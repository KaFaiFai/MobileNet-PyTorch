"""
original preprocessing method is same as Inception, which takes 4×3×6×2 = 144 crops per image
we only take 1 center crop after resizing
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models

from collections import defaultdict
import argparse
from pathlib import Path

from config import *
from script import test_loop
from script.utils import be_deterministic

be_deterministic()

parser = argparse.ArgumentParser()
# required settings
parser.add_argument("--data", type=str, help="dataset", choices=DATASETS.keys(), required=True)
parser.add_argument("--model", type=str, help="model type", choices=MODELS.keys(), required=True)
parser.add_argument("--pretrained-model", type=Path, help="path to pretrained model", required=True)
parser.add_argument("--batch-size", type=int, help="training batch size", required=True)

# misc parameters
parser.add_argument("--device", type=str, help="cpu or gpu?", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--num-workers", type=int, help="sub-processes for data loading", default=0)

# misc settings
parser.add_argument("--print-step", type=int, help="How often to print progress (in batch)?")


def test(configs):
    # hyper parameters
    batch_size = configs.batch_size

    # experiment settings
    data = configs.data
    model_type = configs.model
    device = configs.device if torch.cuda.is_available() else "cpu"
    num_workers = configs.num_workers
    pretrained_model_path = configs.pretrained_model

    assert data in DATASETS.keys()
    assert model_type in MODELS.keys()

    # training configs
    c = defaultdict(lambda: None)
    c["print_step_test"] = configs.print_step
    c["device"] = device

    # select dataset
    Dataset = DATASETS[data]["class"]
    test_root = DATASETS[data]["test_root"]
    test_dataset = Dataset(root=test_root, is_train=False)
    num_class = test_dataset.num_labels

    # set up model, dataloader, optimizer, criterion
    Network = MODELS[model_type]
    network = Network(num_class).to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()

    state = torch.load(pretrained_model_path)
    network.load_state_dict(state["state_dict"])

    # testing loop
    print(f"{'-' * 5} Test result {'-' * 5}")
    test_loop(network, test_dataloader, criterion, **c)


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
