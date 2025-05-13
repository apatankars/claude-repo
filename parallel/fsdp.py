import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torchvision.models import vgg16

import utils

class FullyShardedDataParallel():
    """
    A module implementation of Fully Sharded Data Parallel (FSDP)

    Attributes:
        - layers (nn.ModuleList): a list of all of the layers of the model
        - local_params (dict[int, dict[int, list[nn.Parameter]]): a dictionary 
          mapping the index of each layer to either an empty list if it has no 
          parameters (e.g. ReLU or Flatten), or a list of two parameters 
          (weight and bias)
        - device (torch.device): the device the parts of the model for this
          rank are to be located on
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running FSDP
        - learning_rate (float): the learning rate used for training FSDP
        - optimizer (torch.optim.SGD): the optimizer for only the tensor shards 
          on this worker
        - loss_fn (nn.Module): the loss function used when training this model
    """
    def __init__(self, layers: nn.ModuleList, device: torch.device, 
                 unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]], 
                 learning_rate: float = 0.001, loss_fn: nn.Module = nn.CrossEntropyLoss()):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.layers = layers
        self.unsharded_param_shapes = [[layer_param_data[1] for layer_param_data in layer_params] 
                                       for layer_params in unsharded_param_tensors]
        self.local_params = self.get_local_info(unsharded_param_tensors)
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(
            [param for param_list in self.local_params for param in param_list],
            lr=self.learning_rate
        )
        self.loss_fn = loss_fn
        self.saved = {}
        self.layer_to_input = {}
        self.layer_to_output = {}

    def get_local_info(
        self, unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]]
    ) -> list[list[nn.Parameter]]:
        """
        Shards the given flattened and padded tensors for all parameters in a model, 
        and returning just the local parameter shards for this rank.
        
        Args:
            unsharded_param_tensors (list[list[tuple[torch.Tensor, torch.Size]]]): a 
            list containing, for each layer, either an empty list (if the layer has no 
            parameters) or a list containing the lay's weight and bias *flattened* 
            parameters, each padded to be a multiple of `world_size`, tupled with 
            their *unflattened* sizes
        Returns:
            list[list[nn.Parameter]]: a list containing, for each layer, either
            an empty list (if the layer has no parameters) or a list containing
            this rank's shard of the layer's weight and bias parameters
        """
        local_params = []
        for param_tensors in unsharded_param_tensors:
            layer_params = []
            for param_tensor, _ in param_tensors:
                shard_size = param_tensor.numel() // self.world_size
                layer_params.append(nn.Parameter(data=param_tensor.flatten()[self.rank*shard_size:(self.rank+1)*shard_size]))
            local_params.append(layer_params)
        return local_params

    def gather_param_data(self, layer_idx: int) -> list[torch.Tensor]:
        """
        Given a layer's index within the model's list of layers, returns the 
        full unsharded *unflattened* tensor for that layer's weight and bias 
        parameters by gathering across ranks.
    
        Args:  
            layer_idx (int): the index of the layer whose parameters are to be gathered
        Returns:
             list[torch.Tensor]: a list with zero elements, if the layer has no 
             parameters, or two elements - the layer's gathered weight and bias 
             parameters
        """
        if not self.local_params[layer_idx]:
            return []
        else:
            # First we obtain our local tensors
            local_layer_weights = self.local_params[layer_idx][0]
            local_layer_biases = self.local_params[layer_idx][1]

            # First we need to set up a buffer to store the gathered tensors
            weight_gather= [torch.zeros(local_layer_weights.numel(), device=local_layer_weights.device) for _ in range(self.world_size)] #check maybe using zero(weights.numel())
            bias_gather = [torch.zeros(local_layer_biases.numel(), device=local_layer_biases.device) for _ in range(self.world_size)]

            # Now we gather the tensors
            dist.all_gather(weight_gather, local_layer_weights.flatten())
            dist.all_gather(bias_gather, local_layer_biases.flatten())

            # First we get the shapes of the full tensors
            global_layer_shapes = self.unsharded_param_shapes[layer_idx] # this will be (weight_shape, bias_shape)
            global_weight_shape = global_layer_shapes[0] # this will reconstruct a tensor of the same shape as the original weight
            global_bias_shape = global_layer_shapes[1] # this will reconstruct a tensor of the same shape as the original bias

            num_elements = torch.tensor(global_bias_shape).prod().item()

            global_weight_flat = torch.cat(weight_gather, dim=0)
            global_bias_flat = torch.cat(bias_gather, dim=0)[:num_elements]

            global_weight = global_weight_flat.reshape(global_weight_shape)
            global_bias = global_bias_flat.reshape(global_bias_shape)

            return [global_weight, global_bias]

    def delete_params(self, layer_idx, gathered_params):
        """
        Deletes the parameters of the given layer.
        
        Args:
            layer (torch.Module): the layer whose parameters are to be deleted
        """
        if not self.local_params[layer_idx]:
            return

        self.layers[layer_idx].weight.data = torch.empty(self.unsharded_param_shapes[layer_idx][0])
        self.layers[layer_idx].bias.data = torch.empty(self.unsharded_param_shapes[layer_idx][1])
        del gathered_params

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on the underlying model.
        
        Args:
            input (torch.Tensor): the input to the model
        Returns:
            torch.Tensor: the output of the model
        """
        layer_in = input.to(self.device).requires_grad_(True)
        for layer_idx, layer in enumerate(self.layers):
            # First we gather the parameters for this layer
            params = self.gather_param_data(layer_idx)

            if params:
                layer.weight.data = params[0]
                layer.bias.data = params[1]
            layer_out = layer(layer_in)
            self.saved[layer] = {'input': layer_in, 'output': layer_out}
            layer_in = layer_out

            if params:
                self.delete_params(layer_idx, params)

        return layer_out
        
        # TODO (Task 3.1): Implement!
            
    def get_local_grad_shard(self, param_grad: torch.Tensor) -> torch.Tensor:
        """
        Given the calculated gradient for a parameter, reduces the gradient
        across all ranks, and returns the portion of the *flattened* gradient
        relevant for the stored local shard of that parameter, as averaged
        across all ranks.
        
        Args:
            param_grad (torch.Tensor): the gradient for a parameter
        Returns:
            torch.Tensor: the portion of the given gradient for a parameter, 
            flattened, that is relevant to the local shard of that parameter 
            stored on this rank
        """

        dist.all_reduce(param_grad, op=dist.ReduceOp.SUM)
        param_grad /= self.world_size

        shard_size = param_grad.numel() // self.world_size

        local_grad_shard = param_grad.flatten()[self.rank*shard_size:(self.rank+1)*shard_size]

        return local_grad_shard
    
    def backward(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Performs a backward pass on the underlying model.
        
        Args:
            model_output (torch.Tensor): the output of the model
            target (torch.Tensor): the expected output of the model
        Returns:
            torch.Tensor: the calculated loss of the model's output compared
            to the expected output
        """
        # First we need to calculate the loss
        loss = self.loss_fn(model_output, target)
        upstream_grad = torch.autograd.grad(loss, model_output)[0]

        for layer_idx in reversed(range(len(self.layers))):

            layer = self.layers[layer_idx]
            layer_out = self.saved[layer]['output']
            layer_in = self.saved[layer]['input']
            # gather layer params
            params = self.gather_param_data(layer_idx)
            if params:
                layer.weight.data = params[0]
                layer.bias.data = params[1]

            for param_idx, param in enumerate(layer.parameters()):
                weight_grad = torch.autograd.grad(outputs=layer_out, inputs=param, grad_outputs=upstream_grad, retain_graph=True, allow_unused=True)[0]

                local_grad_shard = self.get_local_grad_shard(weight_grad)

                # Determine the expected size of the local parameter shard
                expected_size = self.local_params[layer_idx][param_idx].numel()
                
                # If the gradient shard is smaller than expected, pad it with zeros
                if local_grad_shard.numel() < expected_size:
                    pad_amount = expected_size - local_grad_shard.numel()
                    padding = torch.zeros(pad_amount, device=local_grad_shard.device)
                    local_grad_shard = torch.cat([local_grad_shard, padding])
                # local_grad_shard.
                if weight_grad is not None:
                    self.local_params[layer_idx][param_idx].grad = local_grad_shard
                    

            upstream_grad = torch.autograd.grad(outputs=layer_out, inputs=layer_in, grad_outputs=upstream_grad, retain_graph=False)[0]
            

            self.delete_params(layer_idx, params)
            
        return loss

    def optimizer_step(self):
        """
        Updates the parameters of the model using the optimizer and accumulated
        gradients. Should be called after each batch's backward pass completes.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # TODO (Task 3.1): Implement!


def train_vgg16_cifar10_fsdp_worker(
    rank: int, world_size: int, train_dataset: Dataset, stats_queue: mp.Queue,
    layers: nn.ModuleList, unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]],
    num_batches: int = 5, cores_per_rank: int = 1, batch_size = 32, 
    learning_rate: float = 1e-2, check_weights: bool = False, check_output: bool = True
):
    """
    For a given worker process, trains a VGG16 model on the CIFAR-10 
    dataset using FSDP.

    Args:
        rank (int): the rank of the current process
        world_size (int): the total number of processes
        train_dataset (Dataset): the training dataset
        stats_queue (mp.Queue): the queue for communicating statistics about this worker
        layers (nn.ModuleList): a list of layers in the model to be trained
        unsharded_param_tensors (list[list[tuple[torch.Tensor, torch.Size]]]): a 
        list containing, for each layer, either an empty list (if the layer has no 
        parameters) or a list containing the layer's weight and bias *flattened* 
        parameters, each padded to be a multiple of `world_size`, tupled with 
        their *unflattened* sizes
        num_batches (int): the number of batches to train for; defaults to 5
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        batch_size (int): the number of data points processed in one step of
        training; defaults to 64
        learning_rate (float): the learning rate of the optimizer; defaults to 1e-2
        check_weights (bool): boolean to determine whether weights are being saved;
        defaults to False
        check_output (bool): boolean to determine whether model output on a single
        batch is saved; defaults to True
    """
    device = torch.device("cpu")
    stats = {
        utils.RANK: rank,
        utils.TOTAL_TIME: 0.0,
    }

    utils.debug_print(f"Initializing FSDP on rank {rank}.")
    utils.parallel_setup(rank, world_size)
    utils.pin_to_core(rank, cores_per_rank)
    utils.seed_everything(1390)

    # set up distributed data loader
    distributed_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=distributed_sampler,
        shuffle=False
    )

    model = FullyShardedDataParallel(layers, device, unsharded_param_tensors, learning_rate)
    
    print(f"Starting training for {num_batches} batches...")

    for i, (inputs, labels) in enumerate(train_loader):
        if i == num_batches:
            stats[utils.TOTAL_TIME] /= num_batches
            break
        start = time.time()
        utils.debug_print(f"Rank {rank} processing batch {i}")
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        outputs = model.forward(inputs)
        fw_time = time.time()
        loss = model.backward(outputs, labels)
        bw_time = time.time()
        model.optimizer_step()
        opt_end = time.time()

        stats[utils.TOTAL_TIME] += opt_end - start

        print(f"Rank {rank} batch {i + 1}/{num_batches}:\n"
            f" - batch loss: {loss.item():.3f}\n"
            f" - fw time: {fw_time - start:.3f} sec\n"
            f" - bw time: {bw_time - fw_time:.3f} sec\n"
            f" - full fw/bw time: {bw_time - start:.3f} sec\n"
            f" - opt update time: {opt_end - bw_time:.3f} sec\n"
            f" - total time: {opt_end - start:.3f} sec") 
    
    if check_weights:
        state_dict = {}
        for i, layer in enumerate(model.layers):
            full_tensors = model.gather_param_data(i)
            if full_tensors:
                state_dict[f'{i}.weight'] = full_tensors[0]
                state_dict[f'{i}.bias'] = full_tensors[1]
        torch.save(state_dict, f'./state_dicts/rank_{rank}_weights.pt')
    if check_output:
        print(f"Saving model output for rank {rank}")
        torch.save(model.forward(inputs), f'./saved_tensors/rank_{rank}_test_output.pt')

    if rank == 0:
        print("Finished training")

    stats_queue.put(stats)
    utils.parallel_cleanup()


def init_layers_and_params(world_size: int) -> tuple[nn.ModuleList, list[list[tuple[torch.Tensor, torch.Size]]]]:
    """
    Loads/initializes the VGG16 model and returns its layers and each layer's 
    padded parameter tensors.
    
    Args:
        world_size (int): the number of worker processes running FSDP
    Returns:
        tuple[nn.ModuleList, list[list[tuple[torch.Tensor, torch.Size]]]: a tuple of:
         - the list of layers in the VGG16 model
         - a list containing, for each layer, either an empty list (if the layer 
           has no parameters) or a list containing the layer's weight and bias 
           *flattened* parameters, each padded to be a multiple of `world_size`, 
           tupled with their *unflattened* sizes
    """
    utils.seed_everything(1390)
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    model.eval()
    
    def pad_and_flatten_param(orig_param_tensor: torch.Tensor) -> torch.Tensor:
        orig_tensor_size = orig_param_tensor.numel()
        if orig_tensor_size % world_size != 0:
            padded_tensor_size = orig_tensor_size + (world_size - (orig_tensor_size % world_size))
            padded_param_tensor = torch.zeros(size=(padded_tensor_size,))
            padded_param_tensor[0:orig_tensor_size] = orig_param_tensor.flatten()
            return padded_param_tensor
        else:
            return orig_param_tensor.flatten()

    layers = nn.ModuleList([layer for layer in model.features] + [model.avgpool, nn.Flatten()] + [layer for layer in model.classifier])
    unsharded_param_tensors: list[list[tuple[torch.Tensor, torch.Size]]] = []  
    for layer in layers:
        if hasattr(layer, 'weight'):
            unsharded_param_tensors.append([
                (pad_and_flatten_param(layer.weight.data), layer.weight.data.shape), 
                (pad_and_flatten_param(layer.bias.data), layer.bias.data.shape)
            ])
        else:
            unsharded_param_tensors.append([])
                
    return layers, unsharded_param_tensors


def train_vgg16_cifar10_fsdp(
    world_size: int = 1, num_batches: int = 5, batch_size: int = 32,
    learning_rate: float = 1e-2, cores_per_rank = 1, 
    check_weights: bool = False, check_output: bool = True
) -> dict:
    """
    Trains a VGG16 model on the CIFAR-10 dataset using FSDP.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_weights (bool): boolean to determine whether weights are being saved;
        defaults to False
        check_output (bool): boolean to determine whether model output on a single
        batch is saved; defaults to True
    """    
    layers, unsharded_param_tensors = init_layers_and_params(world_size)
    utils.debug_print("Created model parameter information")
    
    stats_queue = mp.Queue()
    mp.spawn(
        train_vgg16_cifar10_fsdp_worker,
        args=(world_size, utils.get_train_dataset(), stats_queue, layers, 
              unsharded_param_tensors, num_batches, cores_per_rank, batch_size, 
              learning_rate, check_weights, check_output),
        nprocs=world_size,
        join=True
    )

    return utils.agg_stats_per_rank(stats_queue)
