from dataset import *

DATASETS = {
    "stanford-dogs": {
        "class": DogsDataset,
        "train_root": r"D:\_Dataset\Stanford Dogs",
        "test_root": r"D:\_Dataset\Stanford Dogs",
    }, "mnist": {
        "class": MNISTDataset,
        "train_root": r"D:\_Dataset",
        "test_root": r"D:\_Dataset",
    }, "cifar10": {
        "class": CIFAR10Dataset,
        "train_root": r"D:\_Dataset\CIFAR10",
        "test_root": r"D:\_Dataset\CIFAR10",
    }, "image-net": {
        "class": ImageNetDataset,
        "train_root": r"D:\_Dataset\ImageNet_2012",
        "test_root": r"D:\_Dataset\ImageNet_2012",
    },
}
