"""
Load dataset to see if any error occurs
"""
import timeit

from torch.utils.data import DataLoader
from dataset import *


def load_dataset():
    start = timeit.default_timer()
    # dataset = DogsDataset(root=r"D:\_Dataset\Stanford Dogs", is_train=True)
    # dataset = MNISTDataset(root=r"D:\_Dataset", is_train=True)
    # dataset = CIFAR10Dataset(root=r"D:\_Dataset", is_train=True)
    print(ImageNetDataset.label2name(0))
    dataset = ImageNetDataset(root=r"D:\_Dataset\ImageNet_2012", is_train=True)
    print(dataset.num_labels)
    num_prints = 100
    batch_size = 64

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    last_batch = -1
    mid = timeit.default_timer()
    print(f"Time for reading dataset: {mid - start:.2f}s")

    try:
        for batch_idx, _ in enumerate(dataloader):
            if batch_idx % (len(dataloader) // num_prints) == 0:
                print(f"Loading [Batch {batch_idx:4d}/{len(dataloader)}] ...")
            last_batch = batch_idx
    except Exception as e:
        print(f"Loading batch {last_batch + 1} with batch_size={batch_size} when error occurs")
        print(e)

    end = timeit.default_timer()
    print(f"Time for iterating dataset with batch_size={batch_size}: {end - mid:.2f}s | "
          f"{(end - mid) / len(dataloader):.2f}s/batch")


if __name__ == '__main__':
    load_dataset()
