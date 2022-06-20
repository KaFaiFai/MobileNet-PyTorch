import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tools import ClassificationMetrics


def validate(network: Module, dataloader: DataLoader, criterion: Module, **kwargs):
    device = kwargs["device"]
    num_prints = kwargs["num_prints"]
    num_batches = len(dataloader)
    digits = int(np.log10(num_batches)) + 1  # for print

    total_loss = 0  # BCE loss
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            network.eval()
            outputs = network(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels += labels.tolist()
            all_preds += preds.tolist()

            metrics = ClassificationMetrics(labels, preds)
            if num_prints is not None and batch_idx % (num_batches // num_prints) == 0:
                print(
                    f"[Batch {batch_idx:{digits}d}/{num_batches}] "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {metrics.accuracy:.2%}")

    total_loss /= len(dataloader)
    print(f"Total test data: {len(all_labels)}, Loss: {total_loss:.4f}")
    metrics = ClassificationMetrics(all_labels, all_preds)
    metrics.print_report()
