import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self, num_class, **kwargs):
        super().__init__()
        self.num_class = num_class

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 5000)
        self.fc2 = nn.Linear(5000, num_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __repr__(self):
        return f"LeNet({self.num_class})"
