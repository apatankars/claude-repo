import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torchvision.models import vgg16

import utils

class DistributedDataParallel():
    """
    A module implementation of Distributed Data Parallel (DDP)
    
    Attributes:
        - model (nn.Module): the underlying model to be trained
        - device (torch.device): the device the model is located on for this rank
        - rank (int): the rank of the process the module is located on
        - world_size (int): the total number of processes running DDP
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def broadcast_params(self):
        """
        Broadcasts the underlying model's parameters across all ranks
        """
        # TODO (Task 1.1): Implement!
        for param in self.model.parameters():
            dist.broadcast(param, 0)


    def average_gradients(self):
        """
        Averages the gradients of all model parameters across all ranks.
        """
        # TODO (Task 1.1): Implement!
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size


def train_vgg16_cifar10_ddp_worker(
    rank: int, world_size: int, train_dataset: Dataset, stats_queue: mp.Queue, 
    num_batches: int = 5, cores_per_rank: int = 1, 
    batch_size: int = 32, learning_rate: float = 1e-2,
    check_weights: bool = False, check_output: bool = True
):
    """
    For a given worker process, trains a VGG16 model on the CIFAR-10 
    dataset using DDP.
    
    Args:
        rank (int): the rank of the current process
        world_size (int): the total number of processes
        train_dataset (Dataset): the training dataset
        stats_queue (mp.Queue): the queue for communicating statistics about this worker
        num_batches (int): the number of batches to train for; defaults to 5
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the learning rate of the optimizer; defaults to 1e-2
        check_weights (bool): boolean to determine whether weights are saved;
        defaults to False
        check_output (bool): boolean to determine whether the model's output is
        saved; defaults to True
    """
    device = torch.device("cpu")
    stats = {
        utils.RANK: rank,
        utils.COMP_TIME: 0.0,
        utils.COMM_TIME: 0.0,
        utils.OPT_TIME: 0.0,
        utils.TOTAL_TIME: 0.0,
    }
    
    utils.debug_print(f"Initializing DDP on rank {rank}.")
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

    # load vgg16 model
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    layers = [layer for layer in model.features] + [model.avgpool, nn.Flatten()] + [layer for layer in model.classifier]
    model = nn.Sequential(*layers)
    model = model.to(device)
    model = DistributedDataParallel(model, device)
    model.model.eval()

    # broadcast model
    utils.debug_print(f"Rank {rank}: Broadcasting model to all ranks")
    model.broadcast_params()
    utils.debug_print(f"Rank {rank}: Model finished broadcasting to all ranks")

    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=learning_rate)

    print(f"Starting training for {num_batches} batches...")

    for i, (inputs, labels) in enumerate(train_loader):
        if num_batches == i: # average stats across all batches
            stats[utils.COMP_TIME] /= num_batches
            stats[utils.COMM_TIME] /= num_batches
            stats[utils.OPT_TIME] /= num_batches
            stats[utils.TOTAL_TIME] /= num_batches
            break
        
        utils.debug_print(f"Rank {rank} processing batch {i}")
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        start = time.time()
        outputs = model.model(inputs)

        fw_time = time.time()
        loss = loss_fn(outputs, labels)
        loss.backward()
        bw_time = time.time()
        
        model.average_gradients()

        comm_end = time.time()
        
        optimizer.step()
        opt_end = time.time()

        stats[utils.COMP_TIME] += bw_time - start
        stats[utils.COMM_TIME] += comm_end - bw_time
        stats[utils.OPT_TIME] += opt_end - comm_end
        stats[utils.TOTAL_TIME] += opt_end - start

        print(f"Rank {rank} batch {i + 1}/{num_batches}:\n"
            f" - batch loss: {loss.item():.3f}\n"
            f" - fw time: {fw_time - start:.3f} sec\n"
            f" - bw time: {bw_time - fw_time:.3f} sec\n"
            f" - full fw/bw time: {bw_time - start:.3f} sec\n"
            f" - comm time: {comm_end - bw_time:.3f} sec\n"
            f" - opt update time: {opt_end - comm_end:.3f} sec\n"
            f" - total time: {opt_end - start:.3f} sec")

        
    if check_weights:
        torch.save(model.model.state_dict(), f'./state_dicts/rank_{rank}_weights.pt')
    if check_output:
        print(f"Saving model's output on rank {rank}")
        torch.save(model.model(inputs), f'./saved_tensors/rank_{rank}_test_output.pt')

    # Evaluate the model on the test dataset
    if rank == 0:
        print("Finished training")

    stats_queue.put(stats)
    utils.parallel_cleanup()

def train_vgg16_cifar10_ddp(
    world_size: int = 1, num_batches: int = 5, batch_size: int = 32, 
    learning_rate: float = 1e-2, cores_per_rank: int = 1, 
    check_weights: bool = False, check_output: bool = True
) -> dict:
    """
    Trains a VGG16 model on the CIFAR-10 dataset using DDP.
    
    Args:
        world_size (int): the number of processes to train with; defaults to 1
        num_batches (int): the number of batches to train for; defaults to 5
        batch_size (int): the number of data points processed in one step of
        training; defaults to 32
        learning_rate (float): the optimizer's learning rate; defaults to 1e-2
        cores_per_rank (float): the number of cores to pin to each rank; defaults to 1
        check_weights (bool): boolean to determine whether weights are saved;
        defaults to False
        check_output (bool): boolean to determine whether the model's output is
        saved; defaults to True
    """
    stats_queue = mp.Queue()
    mp.spawn(
        train_vgg16_cifar10_ddp_worker, 
        args=(world_size, utils.get_train_dataset(), stats_queue, num_batches, 
              cores_per_rank, batch_size, learning_rate, check_weights, check_output), 
        nprocs=world_size, join=True
    )
    return utils.agg_stats_per_rank(stats_queue)
