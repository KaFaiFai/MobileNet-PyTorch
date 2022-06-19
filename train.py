from pathlib import Path

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from model import *
from dataset import *
from script import train_epoch, validate
from script.utils import find_next_id, be_deterministic

be_deterministic()


def train():
    # hyper parameters
    batch_size = 64
    lr = 3e-4

    # experiment settings
    data = "imagenet"  # ["dogs", "mnist", "cifar10", "imagenet"]
    model_type = "mobile_net"  # ["mobile_net", "lenet"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2
    num_epochs = 50
    save_step = 1
    out_directory = r".\out"
    pretrained_model_path = r"C:\_Project\Pycharm Projects\MobileNet\out\0004\network_mobile_net_0010.pth"  # r"C:\_Project\Pycharm Projects\MobileNet\out\0001\network_0030.pth"

    assert data in ["dogs", "mnist", "cifar10", "imagenet"]
    assert model_type in ["mobile_net", "lenet"]

    # training configs
    c = dict()
    c["num_prints"] = 5
    c["device"] = device

    out_path = Path(out_directory) / f"{find_next_id(Path(out_directory)):04d}"

    # select dataset
    train_dataset = None
    test_dataset = None
    if data == "dogs":
        train_dataset = DogsDataset(root=r"D:\_Dataset\Stanford Dogs", is_train=True)
        test_dataset = DogsDataset(root=r"D:\_Dataset\Stanford Dogs", is_train=False)
    elif data == "mnist":
        train_dataset = MNISTDataset(root=r"D:\_Dataset", is_train=True)
        test_dataset = MNISTDataset(root=r"D:\_Dataset", is_train=False)
    elif data == "cifar10":
        train_dataset = CIFAR10Dataset(root=r"D:\_Dataset\CIFAR10", is_train=True)
        test_dataset = CIFAR10Dataset(root=r"D:\_Dataset\CIFAR10", is_train=False)
    elif data == "imagenet":
        train_dataset = ImageNetDataset(root=r"D:\_Dataset\ImageNet_2012", is_train=True)
        test_dataset = ImageNetDataset(root=r"D:\_Dataset\ImageNet_2012", is_train=False)

    # set up dataloader, model, optimizer, criterion
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    network = None
    if model_type == "lenet":
        network = LeNet(train_dataset.num_labels).to(device)
    elif model_type == "mobile_net":
        network = MobileNet(train_dataset.num_labels).to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    from_epoch = 0
    if pretrained_model_path is not None:
        state = torch.load(pretrained_model_path)
        network.load_state_dict(state['state_dict'])
        from_epoch = state['epoch'] + 1

    # training loop
    for epoch in range(from_epoch, num_epochs):
        print(f"{'-' * 10} Epoch {epoch:2d}/{num_epochs} {'-' * 10}")
        train_epoch(network, train_dataloader, optimizer, criterion, **c)

        print(f"{'-' * 5} Validation result {'-' * 5}")
        validate(network, test_dataloader, criterion, **c)

        if epoch % save_step == 0:
            out_path.mkdir(exist_ok=True, parents=True)
            save_to = out_path / f"network_{model_type}_{epoch:04d}.pth"
            print(f"{'-' * 5} Saving model to {save_to} {'-' * 5}")
            state = {'epoch': epoch, 'state_dict': network.state_dict()}
            torch.save(state, str(save_to))

        print('\n')


if __name__ == '__main__':
    train()
