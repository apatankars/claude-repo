import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from einops import rearrange

from lm import LMHeadModel
from data import TensorDataLoader

def accuracy_ignore_index(logits: torch.Tensor, y: torch.Tensor, ignore_index: int = -100):
    num_classes = logits.shape[-1]
    preds = torch.argmax(logits, dim=-1)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return torchmetrics.functional.classification.accuracy(
        preds,
        y,
        num_classes=num_classes,
        ignore_index=ignore_index,
        average="micro",
        task="multiclass",
    )


class LMSynthetic:
    def __init__(
        self,
        train_data: TensorDataLoader,
        test_data: TensorDataLoader,
        input_seq_len: int,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        n_layers: int = 4,
        **kwargs,
    ):
        self.input_seq_len = input_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers

        self.trainloader = train_data
        self.testloader = test_data
        self.args = kwargs
        self.device = kwargs["device"]

    def load_model(self):
        self.model = LMHeadModel(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_layers=self.n_layers,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.input_seq_len + 2,
        )
        self.model.to(self.device)

    def train(self, epoch: int):
        self.model.train()
        all_outputs = []
        all_targets = []
        for inputs, targets in self.trainloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # forward
            outputs = self.model(inputs)

            # loss computation
            outputs = outputs[0]
            outputs = rearrange(outputs, "... C -> (...) C")
            targets = rearrange(targets, "... -> (...)")
            all_outputs.append(outputs)
            all_targets.append(targets)

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

        # backprop
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        loss = self.loss_fn(all_outputs, all_targets)
        print(f"Epoch: {epoch} | Loss: {loss.item():.3f}")

    def test(self):
        self.model.eval()
        test_loss = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs[0]
                outputs = rearrange(outputs, "... C -> (...) C")
                targets = rearrange(targets, "... -> (...)")
                all_outputs.append(outputs)
                all_targets.append(targets)
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            sample_size = all_outputs.shape[0] // self.input_seq_len
            loss = self.loss_fn(all_outputs, all_targets)
            test_loss += loss.item()
            test_accuracy = accuracy_ignore_index(all_outputs, all_targets) * 100
            print(f"Test accuracy: {test_accuracy:.3f}%")

    def run(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args["lr"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args["epochs"], eta_min=0.0
        )
        for epoch in range(self.args["epochs"]):
            self.train(epoch + 1)
            self.test()
            self.scheduler.step()
