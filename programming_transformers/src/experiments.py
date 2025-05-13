import argparse
import random
import numpy as np
import torch

from data import ICLDataModule, TensorDataLoader
from task import LMSynthetic

def get_args():
    parser = argparse.ArgumentParser(description="arguments")

    # training configs
    parser.add_argument("-t", "--task", required=True, help="task name")
    parser.add_argument("-s", "--seed", default=0, help="random seed")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="number of epochs")
    parser.add_argument(
        "-lr", "--learning_rate", default=0.0005, type=float, help="learning rate"
    )
    parser.add_argument(
        "-tb", "--train_batch", default=32, type=int, help="training batch size"
    )
    parser.add_argument("-l", "--layers", default=4, type=int, help="number of layers")
    parser.add_argument(
        "-d", "--d_model", default=256, type=int, help="hidden dimension size"
    )

    # dataset
    parser.add_argument(
        "-n_train",
        "--n_train",
        default=10000,
        type=int,
        help="number of training examples",
    )
    parser.add_argument(
        "-n_test", "--n_test", default=500, type=int, help="number of test examples"
    )
    parser.add_argument("-v", "--vocab_size", default=20, type=int, help="vocab size")
    parser.add_argument(
        "-is", "--input_seq_len", default=100, type=int, help="input sequence length"
    )

    args = parser.parse_args()
    return args

def set_determinism(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def load_data(args: argparse.Namespace) -> tuple[TensorDataLoader, TensorDataLoader]:
    dataset = ICLDataModule(
        num_examples=args.n_train,
        num_test_examples=args.n_test,
        vocab_size=args.vocab_size,
        input_seq_len=args.input_seq_len,
        batch_size=args.train_batch,
    )
    dataset.setup()
    return dataset.train_dataloader(), dataset.test_dataloader()

def create_task_instance(args: argparse.Namespace) -> LMSynthetic:
    task_name = args.task
    train_dataloader, test_dataloader = load_data(args)

    assert task_name == "lm_synthetic", f"Task {task_name} not found"
    return LMSynthetic(
        input_seq_len=args.input_seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        device=device,
        task=args.task,
        epochs=args.epochs,
        lr=args.learning_rate,
        train_data=train_dataloader,
        test_data=test_dataloader,
        n_layers=args.layers,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    args = get_args()
    set_determinism(args)
    task = create_task_instance(args)
    task.load_model()
    task.run()
