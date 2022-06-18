import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def validate(network: Module, dataloader: DataLoader, criterion: Module, **kwargs):
    device, num_prints = kwargs["device"], kwargs["num_prints"]

    total_loss = 0
    total_correct = 0
    total_data = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            network.eval()
            outputs = network(images)
            loss = criterion(outputs, labels)

            total_data += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()

    print(f"Total test data: {total_data}, "
          f"Loss: {total_loss / total_data:.4f}, "
          f"Accuracy: {total_correct / total_data:.2%}")
