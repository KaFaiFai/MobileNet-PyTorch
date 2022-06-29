"""
random stuff goes here
"""
from collections import defaultdict
from pathlib import Path
import torch
from torch import Tensor
from torch import nn
import tensorflow as tf
import numpy as np
import random


def find_next_id(path: Path):
    folder_names = [p.name for p in Path(path).glob("*/")]
    all_ids = []
    for n in folder_names:
        try:
            n_int = int(n)
            all_ids.append(n_int)
        except:
            pass

    if len(all_ids) == 0:
        next_id = 0
    else:
        next_id = max(all_ids) + 1

    return next_id


def gray2rgb(image: Tensor):
    return image.repeat(3, 1, 1)


def be_deterministic(seed=0):
    # may not work
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def return_none(*args, **kwargs):
    return None


def to_nearest_multiple_of(num, multiple):
    return round(num / multiple) * multiple


def defaultdict_none(init_dict=None, **kwargs):
    if init_dict is None:
        init_dict = dict()
    whole_dict = {**init_dict, **kwargs}
    return defaultdict(return_none, **whole_dict)


def peek_pytorch_network(network: nn.Module):
    network_state = network.state_dict()
    print("PyTorch model's state_dict:")
    for layer_name, tensor in network_state.items():
        print(f"{layer_name:<60}: {tensor.size()}")


def peek_tensorflow_network(network: tf.keras.Model):
    for layer in network.layers:
        if len(layer.get_weights()) > 0:
            for t, w in zip(layer.weights, layer.get_weights()):
                print(f"{t.name:<60}: {w.shape}")


def test():
    # next_id = find_next_id(Path(r"C:\_Project\Pycharm Projects\MobileNet\out"))
    # print(next_id)
    init_dict = {"a": 1, "b": 2}
    print(defaultdict_none(init_dict, c=3, b=4))


if __name__ == '__main__':
    test()
