import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torch.distributed as dist
import torch.multiprocessing as mp

import utils

class ModelParallelWorker():
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
        - world_size (int): the total number of processes running model parallelism
        - learning_rate (float): the learning rate of the optimizer
        - optimizer (torch.optim.Adam): the optimizer for training this part of
          the model
    """
    def __init__(self, model_part: nn.Sequential, device: torch.device,
                 batch_times: list[float], learning_rate: float = 0.001):
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
        self.saved = {}
        self.saved_input = None
        self.saved_output = None

    def forward(self, input_size: torch.Size, input: torch.Tensor | None) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            input_size (torch.Size): the size of the input tensor
            input (torch.Tensor | None): the input tensor to the model if this 
            process has the lowest rank; otherwise None
        Returns:
            torch.Tensor: the output tensor of this part of the model
        """
        x = torch.empty(input_size, device=self.device, requires_grad=True)
        if self.rank == 0:
            assert input is not None
            x = input.to(self.device).requires_grad_()
        else:
            dist.recv(x, src=self.rank - 1)  # Ensure the received tensor tracks gradients
            
        utils.debug_print(f"Rank {self.rank}: starting forward pass")
        self.batch_times.append(time.time())

        for layer in self.model_part:

            layer_out = layer(x)
            self.saved[layer] = {"input": x, "output": layer_out}
            x = layer_out

        self.batch_times.append(time.time())
        utils.debug_print(f"Rank {self.rank}: finished forward pass")

        # Only send the output if there is another GPU or else infinite hang
        if self.rank != self.world_size - 1:
            dist.send(x, self.rank + 1)
        return x

    def backward(self, grad_size: torch.Size, grad: torch.Tensor | None):
        """
        Performs the backward pass of the model.

        Args:
            grad_size (torch.Size): the size of the gradient tensor
            grad (torch.Tensor | None): the gradient tensor of loss with respect to
            the model's output if this process has the highest rank; otherwise None
        """
        upstream_grad = torch.empty(grad_size, device=self.device)
        if self.rank == self.world_size - 1:
            assert grad is not None
            upstream_grad = grad.to(self.device)
        else:
            dist.recv(upstream_grad, src=self.rank + 1)

        utils.debug_print(f"Rank {self.rank}: starting backward pass")
        self.batch_times.append(time.time())

        # Compute the gradient for each layer in reverse order
        for layer in reversed(self.model_part):
            layer_out = self.saved[layer]["output"]
            layer_in = self.saved[layer]["input"]
            params = list(layer.parameters())

            if params:
                weight_grad = torch.autograd.grad(
                    outputs=layer_out,
                    inputs=params,
                    grad_outputs=upstream_grad,
                    retain_graph=True
                )
                for p, g in zip(params, weight_grad):
                    p.grad = g
            # Update upstream_grad via the layer's input regardless of parameters
            upstream_grad = torch.autograd.grad(
                outputs=layer_out,
                inputs=layer_in,
                grad_outputs=upstream_grad,
                retain_graph=False
            )[0]

        self.batch_times.append(time.time())
        utils.debug_print(f"Rank {self.rank}: finished backward pass")

        self.optimizer_step()

        if self.rank != 0:
            dist.send(upstream_grad, self.rank - 1)
        return upstream_grad
    
    def optimizer_step(self):
        """
        Updates the parameters of these layers using the optimizer and accumulated
        gradients. Should be called after each batch's backward pass completes.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

class ModelParallel():
    """
    A module implementation of model parallelism.
    
    Attributes:
        - device (torch.device): the device for this model to reside on
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running model parallelism
        - worker (ModelParallelWorker): the underlying class responsible for this
          rank's partitioned part of the model
        - loss_fn (nn.Module): the loss function to use when training
        - part_comm_info (dict[str, torch.Size]): the results of communication 
          analysis, storing the activation size and gradient size for this partition
    """
    def __init__(self, partition: list[nn.Module], device: torch.device, 
                 part_comm_info: dict[str, torch.Size],
                 batch_times: list[float], learning_rate: float = 0.001, 
                 loss_fn: nn.Module = nn.CrossEntropyLoss()):
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.worker = ModelParallelWorker(
            nn.Sequential(*partition), device, batch_times, learning_rate
        )
        self.loss_fn = loss_fn
        self.part_comm_info = part_comm_info

    def forward(self, input_tensor: torch.Tensor | None) -> torch.Tensor:
        """
        Performs the forward pass of the model for this rank.

        Args:
            input_tensor (torch.Tensor | None): the input tensor to the full model,
            if this rank is responsible for the first layers of the model;
            otherwise None
        Returns:
            torch.Tensor: the output tensor of the model for this rank's layers
        """
        input = None
        if self.rank == 0:
            assert input_tensor is not None
            input = input_tensor
        
        return self.worker.forward(input_size=self.part_comm_info["activation_size"], input=input)
        
        # TODO (Task 2.2): Implement!

    def backward(self, model_output: torch.Tensor | None, target: torch.Tensor | None):
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
        """
        grad_output = None
        if self.rank == self.world_size - 1:
            assert model_output is not None
            assert target is not None
            loss = self.loss_fn(model_output, target)
            grad_output = torch.autograd.grad(loss, model_output, retain_graph=False)[0]
            # Receive the gradient from the next rank.
        self.worker.backward(grad_size=self.part_comm_info["gradient_size"], grad=grad_output)

    def train_step(self, input_tensor: torch.Tensor, target: torch.Tensor):
        """
        Performs a training step for the model for this rank (i.e. for this 
        rank's layers).

        Args:
            input_tensor (torch.Tensor): the input tensor to the full model
            target (torch.Tensor): the expected output tensor of the full model
        """
        
        output = self.forward(input_tensor=input_tensor)
        self.backward(model_output=output, target=target)
        
    
def train_vgg16_cifar10_model_parallel_worker(
    rank: int, world_size: int, partitions: list[list[nn.Module]],
    communication_info: list[dict[str, torch.Size]], train_loader: DataLoader,
    stats_queue: mp.Queue, num_batches: int = 5, cores_per_rank: int = 1, 
    learning_rate: float = 1e-2, check_output: bool = True
):
    """
    For a given worker process, trains a VGG16 model on the CIFAR-10 dataset 
    using model parallelism.

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
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        learning_rate (float): the learning rate of the optimizer; defaults to 1e-2
        check_output (bool): boolean to determine whether to save the model's output;
        defaults to True
    """
    device = torch.device("cpu")
    stats = {
        utils.RANK: rank,
        utils.TOTAL_TIME: 0.0,
        utils.BATCHES_TIMES: []
    }
    
    utils.debug_print(f"Initializing model parallelism on rank {rank}.")
    utils.parallel_setup(rank, world_size)
    utils.pin_to_core(rank, cores_per_rank)
    utils.seed_everything(1390)

    model_parallel_wrapper = ModelParallel(partitions[rank], device, communication_info[rank], 
                                           stats[utils.BATCHES_TIMES], learning_rate)
    for i, (inputs, labels) in enumerate(train_loader):
        if num_batches == i: # average stats across all batches
            stats[utils.TOTAL_TIME] /= num_batches
            break
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        model_parallel_wrapper.train_step(inputs, labels)
        worker_end = time.time()
        
        # barrier for all workers to finish the current batch
        dist.barrier()
        end = time.time()
        stats[utils.TOTAL_TIME] += end - start
        print(
            f"Rank {rank} batch {i + 1}/{num_batches}:\n"
            f" - worker time: {worker_end - start:.3f} sec\n"
            f" - total time: {end - start:.3f} sec"
        )

    if check_output:
        test_output = model_parallel_wrapper.forward(inputs)
        if rank == world_size - 1:
            print("Saving model output")
            torch.save(test_output, f'./saved_tensors/rank_{rank}_test_output.pt')
    stats_queue.put(stats)
    utils.parallel_cleanup()

def train_vgg16_cifar10_model_parallel(
    world_size: int = 1, num_batches: int = 5, batch_size: int = 32,
    learning_rate: float = 1e-2, cores_per_rank: int = 1, check_output: bool = True
) -> dict:
    """
    Trains and evaluates a VGG16 model on the CIFAR-10 dataset using model parallelism.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_output (bool): boolean to determine whether to save the model's
        output; defaults to True
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
    
    communication_info = utils.analyze_communication_with_partitions(
        next(iter(train_loader))[0], partitions
    )
    
    stats_queue = mp.Queue()
    mp.spawn(
        train_vgg16_cifar10_model_parallel_worker,
        args=(world_size, partitions, communication_info, train_loader, 
              stats_queue, num_batches, cores_per_rank, learning_rate, check_output),
        nprocs=world_size,
        join=True
    )
    return utils.agg_stats_per_rank(stats_queue)
