from pathlib import Path
import argparse
from collections import defaultdict

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from config import *
from script import train_epoch, test_loop
from script.utils import find_next_id, be_deterministic

be_deterministic()

parser = argparse.ArgumentParser()
# required settings
parser.add_argument("--data", type=str, help="dataset", choices=DATASETS.keys(), required=True)
parser.add_argument("--model", type=str, help="model type", choices=MODELS.keys(), required=True)
parser.add_argument("--batch-size", type=int, help="training batch size", required=True)

# hyper parameters
parser.add_argument("--lr", type=float, help="training learning rate", default=3e-4)
parser.add_argument("--alpha", type=float, help="width multiplier of MobileNet", default=1.0)
parser.add_argument("--input-resolution", type=int, help="input resolution of MobileNet", default=224)

# misc parameters
parser.add_argument("--device", type=str, help="cpu or gpu?", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--num-workers", type=int, help="sub-processes for data loading", default=0)
parser.add_argument("--num-epochs", type=int, help="training epochs", default=50)

# misc settings
parser.add_argument("--print-step", type=int, help="How often to print progress (in batch)?")
parser.add_argument("--save-step", type=int, help="How often to save network (in epoch)?")
parser.add_argument("--out-dir", type=Path, help="Where to save network (in epoch)?")
parser.add_argument("--resume", type=Path, help="path to saved network")


def train(configs):
    # hyper parameters
    batch_size = configs.batch_size
    lr = configs.lr
    alpha = configs.alpha
    input_resolution = configs.input_resolution

    # experiment settings
    data = configs.data
    model_type = configs.model
    device = configs.device if torch.cuda.is_available() else "cpu"
    num_workers = configs.num_workers
    num_epochs = configs.num_epochs
    save_step = configs.save_step
    out_directory = configs.out_dir
    pretrained_model_path = configs.resume

    assert data in DATASETS.keys()
    assert model_type in MODELS.keys()

    # training configs
    c = defaultdict(lambda: None)
    c["print_step_train"] = configs.print_step
    c["device"] = device

    # select dataset
    Dataset = DATASETS[data]["class"]
    train_root = DATASETS[data]["train_root"]
    test_root = DATASETS[data]["test_root"]
    train_dataset = Dataset(root=train_root, is_train=True)
    test_dataset = Dataset(root=test_root, is_train=False)
    num_class = train_dataset.num_labels

    # set up model, dataloader, optimizer, criterion
    Network = MODELS[model_type]
    network = Network(num_class).to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    from_epoch = 0
    if pretrained_model_path is not None:
        state = torch.load(pretrained_model_path)
        network.load_state_dict(state["state_dict"])
        from_epoch = max(state["epoch"] + 1, 0)

    # training loop
    for epoch in range(from_epoch, num_epochs):
        print(f"{'-' * 10} Epoch {epoch:2d}/{num_epochs} {'-' * 10}")
        train_epoch(network, train_dataloader, optimizer, criterion, **c)

        print(f"{'-' * 5} Validation result {'-' * 5}")
        test_loop(network, test_dataloader, criterion, **c)

        if save_step is not None and out_directory is not None and epoch % save_step == 0:
            out_path = Path(out_directory) / f"{find_next_id(Path(out_directory)):04d}"
            out_path.mkdir(exist_ok=True, parents=True)
            save_to = out_path / f"{model_type}-a{alpha * 100:3d}-r{input_resolution:d}-c{num_class}-e{epoch:04d}.pth"
            print(f"{'-' * 5} Saving model to {save_to} {'-' * 5}")
            state = {"epoch": epoch, "alpha": alpha, "input_resolution": input_resolution,
                     "num_class": num_class, "state_dict": network.state_dict()}
            torch.save(state, str(save_to))

        print('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
