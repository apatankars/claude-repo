import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torch.distributed as dist
import torch.multiprocessing as mp

import utils

class PipelineParallelWorker():
    """
    A worker class that provides methods for training a set of layers 
    within a model that it is responsible for, via communication with other
    ranks containing the preceding and succeeding set of layers of the model.
    
    Attributes:
        - model_part (nn.Sequential): the set of layers of the model that this worker
         is responsible for
        - device (torch.device): the device for this model part to reside on
        - batch_times (list[float]): the sequential list of times of when a forward
          pass starts and ends, and when a backward pass starts and ends
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running pipeline parallelism
        - learning_rate (float): the learning rate of the optimizer
        - optimizer (torch.optim.Adam): the optimizer for training the part of
          the model
        - num_microbatches (int): the number of microbatches to split each batch into
    """
    def __init__(self, model_part: nn.Sequential, device: torch.device,
                 batch_times: list[float], learning_rate: float = 0.001,
                 num_microbatches: int = 2):
        self.model_part = model_part
        self.device = device 
        self.model_part.to(self.device)
        self.model_part.eval()
        self.batch_times = batch_times
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(
            params=(p for layer in self.model_part for p in layer.parameters()),
            lr=self.learning_rate
        )
        self.num_microbatches = num_microbatches
        self.saved = {}
        # self.saved_input = None
        # self.saved_output = None

    def forward(self, input_size: torch.Size, input: torch.Tensor | None,
                microbatch_idx: int) -> torch.Tensor:
        """
        Performs the forward pass of the model.
        For the first worker, the input tensor is passed as an argument; for the 
        remaining workers, the input tensor is received from the previous worker.
        Stores activations at each layer, per microbatch, for later computing 
        the gradients.

        Args:
            input_size (torch.Size): the size of the input tensor
            input (torch.Tensor | None): the input tensor to the model if this 
            process has the lowest rank; otherwise None
            microbatch_idx (int): the ID of current microbatch within a given batch
        Returns:
            torch.Tensor: the output tensor of this part of the model
        """
        layer_in = torch.empty(input_size, dtype=torch.float32, requires_grad=True).to(self.device)
        if self.rank == 0:
            assert input is not None
            layer_in = input.requires_grad_(True).to(self.device)
        else:
            dist.recv(layer_in, src=self.rank - 1)
        
        utils.debug_print(f"Rank {self.rank}, microbatch {microbatch_idx}: starting forward pass")
        self.batch_times.append(time.time())
        if microbatch_idx not in self.saved:
            self.saved[microbatch_idx] = {}

        for layer in self.model_part:
            layer_out = layer(layer_in)
            self.saved[microbatch_idx][layer] = {'input': layer_in, 'output': layer_out}
            layer_in = layer_out
        
        self.batch_times.append(time.time())
        utils.debug_print(f"Rank {self.rank}, microbatch {microbatch_idx}: finished forward pass")

        if self.rank != self.world_size - 1:
            dist.send(layer_out, dst=self.rank + 1)
        return layer_out

    def backward(self, grad_size: torch.Size, grad: torch.Tensor | None,
                 microbatch_idx: int):
        """
        Performs the backward pass of the model.
        For the last worker, the gradient tensor is passed as an argument; for 
        remaining workers, the gradient tensor is received from the next worker.

        Args:
            grad_size (torch.Size): the size of the gradient tensor
            grad (torch.Tensor | None): the gradient tensor of loss with respect to
            the model's output if this process has the highest rank; otherwise None
            microbatch_idx (int): the ID of current microbatch within a given batch
        """
        upstream_grad = torch.empty(grad_size, dtype=torch.float32, requires_grad=True).to(self.device)
        if self.rank == self.world_size - 1:
            assert grad is not None
            upstream_grad = grad.to(self.device)
        else:
            dist.recv(upstream_grad, src=self.rank + 1)
        
        utils.debug_print(f"Rank {self.rank}, microbatch {microbatch_idx}: starting backward pass")
        self.batch_times.append(time.time())

        for layer in reversed(self.model_part):
            layer_out = self.saved[microbatch_idx][layer]['output']
            layer_in = self.saved[microbatch_idx][layer]['input']

            layer_params = list(layer.parameters())

            if layer_params:
                weight_grad = torch.autograd.grad(outputs=layer_out, inputs=layer_params, grad_outputs=upstream_grad, retain_graph=True)
                for param_idx, param in enumerate(layer_params): 
                    if param.grad is None:
                        param.grad = weight_grad[param_idx] / self.num_microbatches
                    else:
                        param.grad += weight_grad[param_idx] / self.num_microbatches

            upstream_grad = torch.autograd.grad(outputs=layer_out, inputs=layer_in, grad_outputs=upstream_grad, retain_graph=True)[0]
        
        self.batch_times.append(time.time())

        utils.debug_print(f"Rank {self.rank}, microbatch {microbatch_idx}: finished backward pass")
        if self.rank != 0:
            dist.send(upstream_grad, dst=self.rank - 1)
        return upstream_grad
        
        # TODO (Task 4.1): Process the produced gradient for this part of the model
    
    def optimizer_step(self):
        """
        Updates the parameters of these layers using the optimizer and accumulated
        gradients. Should be called after each batch's backward pass completes.
        """
        # TODO (Task 4.1): Implement!
        self.optimizer.step()
        self.optimizer.zero_grad()


class PipelineParallel():
    """
    A module implementation of pipeline parallelism.
    
    Attributes:
        - device (torch.device): the device for this model to reside on
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running model parallelism
        - worker (ModelParallelWorker): the underlying class responsible for this
          rank's partitioned part of the model
        - loss_fn (nn.Module): the loss function to use when training
        - part_comm_info (dict[str, torch.Size]): the results of communication 
          analysis, storing the activation size and gradient size for this partition
        - num_microbatches (int): the number of microbatches to split each batch into
    """
    def __init__(self, partition: list[nn.Module],
                 device: torch.device, part_comm_info: dict[str, torch.Size],
                 batch_times: list[float], learning_rate: float = 0.001, 
                 loss_fn: nn.Module = nn.CrossEntropyLoss(), num_microbatches: int = 2):
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.worker = PipelineParallelWorker(
            nn.Sequential(*partition), device, batch_times, 
            learning_rate, num_microbatches
        )
        self.loss_fn = loss_fn
        self.part_comm_info = part_comm_info
        self.num_microbatches = num_microbatches

    def forward(self, input_tensor: torch.Tensor | None, 
                microbatch_idx: int) -> torch.Tensor:
        """
        Performs the forward pass of the model for this rank.

        Args:
            input_tensor (torch.Tensor | None): the input tensor to the full model,
            if this rank is responsible for the first layers of the model;
            otherwise None
            microbatch_idx (int): the ID of current microbatch within a given batch
        Returns:
            torch.Tensor: the output tensor of the model for this rank's layers
        """
        if self.rank == 0:
            assert input_tensor is not None
            input_tensor = input_tensor.to(self.device)
            model_output = self.worker.forward(input_tensor.shape, input_tensor, microbatch_idx)
        else:
            model_output = self.worker.forward(input_size=self.part_comm_info[utils.ACTIVATION_SIZE], input=None, microbatch_idx=microbatch_idx)
        return model_output


    def backward(self, model_output: torch.Tensor | None, target: torch.Tensor | None,
                 microbatch_idx: int):
        """
        Performs the backward pass of the model, with respect to this module's loss
        function, for this rank.

        Args:
            model_output (torch.Tensor): the output tensor of the full model, 
            if this rank is responsible for the last layers of the model; 
            otherwise None
            target (torch.Tensor): the expected output tensor of the full model,
            if this rank is responsible for the last layers of the model; 
            otherwise None
            microbatch_idx (int): the ID of current microbatch within a given batch
        """
        if self.rank == self.world_size - 1:
            assert model_output is not None
            assert target is not None
            loss = self.loss_fn(model_output, target)
            grad_output = torch.autograd.grad(loss, model_output)[0]
            self.worker.backward(grad_output.shape, grad_output, microbatch_idx)
        else:
            self.worker.backward(self.part_comm_info[utils.GRADIENT_SIZE], None, microbatch_idx)

    def train_step(self, input_tensor: torch.Tensor, target: torch.Tensor):
        """
        Performs a training step for the model for this rank (i.e. for this 
        rank's layers).

        Args:
            input_tensor (torch.Tensor): the input tensor to the full model
            target (torch.Tensor): the expected output tensor of the full model
        """
        input_batches = torch.chunk(input_tensor, self.num_microbatches)
        target_batches = torch.chunk(target, self.num_microbatches)

        outputs = []
        for microbatch_idx in range(self.num_microbatches):
            outputs.append(self.forward(input_batches[microbatch_idx], microbatch_idx))

        for microbatch_idx in reversed(range(self.num_microbatches)):
            self.backward(outputs[microbatch_idx], target_batches[microbatch_idx], microbatch_idx)

        self.worker.optimizer_step()

            # forward pass
        # TODO (Task 4.2): Implement!
    
    def eval(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model on the given input.
        
        Args:
            input (torch.Tensor): the input to the model
        Returns:
            torch.Tensor: the output of the model
        """
        outputs = []

        input_batches = torch.chunk(input, self.num_microbatches)

        for micro_idx, micro_batch in enumerate(input_batches):
            micro_output = self.forward(micro_batch, microbatch_idx=micro_idx)
            if self.rank == self.world_size - 1:
                    outputs.append(micro_output)

        # If we're not the last rank, we return None
        if self.rank == self.world_size - 1:
            return torch.cat(outputs, dim=0)
        else:
            return None

def train_vgg16_cifar10_pipeline_parallel_worker(
    rank: int, world_size: int, partitions: list[list[nn.Module]],
    communication_info: list[dict[str, torch.Size]], train_loader: DataLoader,
    stats_queue: mp.Queue, num_batches: int = 5, num_microbatches: int = 2,
    cores_per_rank: int = 1, learning_rate: float = 1e-2, check_output: bool = True
):
    """
    For a given worker process, trains a VGG16 model on the CIFAR-10 dataset 
    using pipeline parallelism.

    Args:
        rank (int): the rank of this process
        world_size (int): the total number of processes
        partitions (list[list[nn.Module]]): the partitions (a list of layers) of 
        the model for each worker
        communication_info (list[dict[str, torch.Size]]): metadata regarding communication 
        between processes, storing the received activation size and gradient 
        size for each partition
        train_loader (DataLoader): the data loader for the training dataset
        stats_queue (mp.Queue): the queue for communicating statistics about this worker
        num_batches (int): the number of batches to train for; defaults to 5
        num_microbatches (int): the number of microbatches to split each batch into;
        defaults to 2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        learning_rate (float): the learning rate of the optimizer; defaults to 1e-2
        check_output (bool): boolean that determines whether the model's output is
        saved; defaults to True
    """
    device = torch.device("cpu")
    stats = {
        utils.RANK: rank,
        utils.TOTAL_TIME: 0.0,
        utils.BATCHES_TIMES: []
    }
    
    utils.debug_print(f"Initializing pipeline parallelism on rank {rank}.")
    utils.parallel_setup(rank, world_size)
    utils.pin_to_core(rank, cores_per_rank)
    utils.seed_everything(1390)

    pipeline_parallel_wrapper = PipelineParallel(partitions[rank], device, communication_info[rank], 
                                                 stats[utils.BATCHES_TIMES], learning_rate,
                                                 num_microbatches=num_microbatches)
    for i, (inputs, labels) in enumerate(train_loader):
        if num_batches == i: # average stats across all batches
            stats[utils.TOTAL_TIME] /= num_batches
            break
        start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        pipeline_parallel_wrapper.train_step(inputs, labels)

        # barrier for all workers to finish the current batch
        worker_end = time.time()
        dist.barrier()
        end = time.time()
        stats[utils.TOTAL_TIME] += end - start
        print(
            f"Rank {rank} batch {i + 1}/{num_batches}:\n"
            f" - worker time: {worker_end - start:.3f} sec\n"
            f" - total time: {end - start:.3f} sec"
        )

    if check_output:
        test_output = pipeline_parallel_wrapper.eval(inputs)
        if rank == world_size - 1:
            print("Saving model output")
            torch.save(test_output, f'./saved_tensors/rank_{rank}_test_output.pt')

    stats_queue.put(stats)
    utils.parallel_cleanup()

def train_vgg16_cifar10_pipeline_parallel(
    world_size: int = 1, num_batches: int = 5, batch_size: int = 32, 
    num_microbatches: int = 2, learning_rate: float = 1e-2, 
    cores_per_rank: int = 1, check_output: bool = True
) -> dict:
    """
    Trains and evaluates a VGG16 model on the CIFAR-10 dataset using 
    pipeline parallelism.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        num_microbatches (int): the number of microbatches to split each batch into;
        defaults to 2
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_output (bool): boolean that determines whether the model's output is
        saved;defaults to True
    """
    train_dataset = utils.get_train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    utils.seed_everything(1390) 
    model = vgg16(weights=None)
    # because of size change for dataset
    model.classifier[0] = nn.Linear(512, 4096)
    model.classifier[6] = nn.Linear(4096, 10)
    
    partitions = utils.split_vgg16(model, world_size)
    assert(len(partitions) == world_size)
    
    assert batch_size % num_microbatches == 0
    microbatch_size = batch_size // num_microbatches
    communication_info = utils.analyze_communication_with_partitions(
        next(iter(train_loader))[0][0:microbatch_size], partitions
    )
    
    stats_queue = mp.Queue()
    mp.spawn(
        train_vgg16_cifar10_pipeline_parallel_worker,
        args=(world_size, partitions, communication_info, train_loader, 
              stats_queue, num_batches, num_microbatches, cores_per_rank, 
              learning_rate, check_output),
        nprocs=world_size,
        join=True
    )
    return utils.agg_stats_per_rank(stats_queue)
