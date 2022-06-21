"""
original preprocessing method is same as Inception, which takes 4×3×6×2 = 144 crops per image
we only take 1 center crop after resizing
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
from collections import defaultdict

from model import *
from dataset import *
from script import test_loop
from script.utils import be_deterministic

be_deterministic()


def test():
    # hyper parameters
    batch_size = 64

    # experiment settings
    data = "imagenet"  # ["stanford-dogs", "mnist", "cifar10", "imagenet"]
    model_type = "mobile_net"  # ["mobile_net", "lenet"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    pretrained_model_path = r"C:\_Project\Pycharm Projects\MobileNet\pretrained\wjc-mobilenet-a100-r224-c1000-e0000.pth"

    assert data in ["dogs", "mnist", "cifar10", "imagenet"]
    assert model_type in ["mobile_net", "lenet"]

    # training configs
    c = defaultdict(lambda: None)
    c["print_step_test"] = 20
    c["device"] = device

    # select dataset
    test_dataset = None
    if data == "stanford-dogs":
        test_dataset = DogsDataset(root=r"D:\_Dataset\Stanford Dogs", is_train=False)
    elif data == "mnist":
        test_dataset = MNISTDataset(root=r"D:\_Dataset", is_train=False)
    elif data == "cifar10":
        test_dataset = CIFAR10Dataset(root=r"D:\_Dataset\CIFAR10", is_train=False)
    elif data == "imagenet":
        test_dataset = ImageNetDataset(root=r"D:\_Dataset\ImageNet_2012", is_train=False)
    num_class = test_dataset.num_labels

    # set up model, dataloader, optimizer, criterion
    network = None
    if model_type == "lenet":
        network = LeNet(num_class).to(device)
    elif model_type == "mobile_net":
        network = MobileNet(num_class).to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()

    state = torch.load(pretrained_model_path)
    network.load_state_dict(state["state_dict"])

    network = models.mobilenet_v2(pretrained=True)
    network = network.to(device)

    # testing loop
    print(f"{'-' * 5} Test result {'-' * 5}")
    test_loop(network, test_dataloader, criterion, **c)


if __name__ == '__main__':
    test()
