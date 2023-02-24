import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import numpy as np
import timeit

from tools import ClassificationMetrics


def train_loop(network: Module, dataloader: DataLoader, optimizer: Optimizer, criterion: Module, **kwargs):
    start = timeit.default_timer()
    device, print_step_train = kwargs["device"], kwargs["print_step_train"]
    num_batches = len(dataloader)
    digits = int(np.log10(num_batches)) + 1  # for print

    total_loss = 0  # BCE loss
    all_labels = []
    all_outputs = []
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        network.train()
        outputs = network(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        network.eval()
        total_loss += loss.item()
        all_labels += labels.tolist()
        all_outputs += outputs.tolist()

        if print_step_train is not None and batch_idx % print_step_train == 0:
            metrics = ClassificationMetrics(labels, outputs)
            print(f"[Batch {batch_idx:{digits}d}/{num_batches}] "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {metrics.accuracy:.2%}")

    end = timeit.default_timer()
    print(f"Time spent: {end - start:.2f}s | {(end - start) / num_batches:.2f}s/batch")

    total_loss /= len(dataloader)
    print(f"Total train data: {len(all_labels)}, Loss: {total_loss:.4f}")
    metrics = ClassificationMetrics(all_labels, all_outputs)
    metrics.print_report()
