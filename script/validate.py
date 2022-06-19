import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from tools import ClassificationMetrics


def validate(network: Module, dataloader: DataLoader, criterion: Module, **kwargs):
    device = kwargs["device"]

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

    total_loss /= len(dataloader)
    print(f"Total test data: {len(all_labels)}, Loss: {total_loss:.4f}")
    metrics = ClassificationMetrics(all_labels, all_preds)
    metrics.print_report()
