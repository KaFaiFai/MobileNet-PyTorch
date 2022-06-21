import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import numpy as np
import timeit

from tools import ClassificationMetrics


def train_epoch(network: Module, dataloader: DataLoader, optimizer: Optimizer, criterion: Module, **kwargs):
    start = timeit.default_timer()
    device, print_step_train = kwargs["device"], kwargs["print_step_train"]
    num_batches = len(dataloader)
    digits = int(np.log10(num_batches)) + 1  # for print

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        network.train()
        outputs = network(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        network.eval()
        metrics = ClassificationMetrics(labels, outputs)
        if print_step_train is not None and batch_idx % print_step_train == 0:
            print(
                f"[Batch {batch_idx:{digits}d}/{num_batches}] "
                f"Loss: {loss.item():.4f}, "
                f"Accuracy: {metrics.accuracy:.2%}")

    end = timeit.default_timer()
    print(f"Time spent: {end - start:.2f}s | {(end - start) / num_batches:.2f}s/batch")
