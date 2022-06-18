import torch
from torch import nn


class DepthWiseConv(nn.Module):
    def __init__(self, num_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              groups=num_channels)

    def forward(self, x):
        # (B, C, H, W) -> (B, C, H//stride, W//stride)
        x = self.conv(x)
        return x


class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # (B, Cin, H, W) -> (B, Cout, H, W)
        x = self.conv(x)
        return x


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=1):
        super().__init__()
        self.dw_conv = DepthWiseConv(in_channels, stride=downscale)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.pw_conv = PointWiseConv(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # (B, Cin, H, W) -> (B, Cout, H/downscale, W/downscale)
        x = self.relu1(self.bn1(self.dw_conv(x)))
        x = self.relu2(self.bn2(self.pw_conv(x)))
        return x


class MobileNet(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        num_channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        dimensions = [112, 112, 56, 56, 28, 28, 14, 14, 14, 14, 14, 14, 7, 7]
        assert len(num_channels) == len(dimensions)

        downscale = 2
        self.initial = nn.Sequential(
            nn.AdaptiveAvgPool2d(dimensions[0] * downscale),
            nn.Conv2d(3, num_channels[0], kernel_size=3, stride=downscale, padding=1),
        )

        separable_convs = []
        for in_channels, out_channels, in_dim, out_dim in zip(num_channels[:-1], num_channels[1:], dimensions[:-1],
                                                              dimensions[1:]):
            assert (in_dim % out_dim) == 0
            separable_convs.append(SeparableConv(in_channels, out_channels, downscale=in_dim // out_dim))
        self.separable_convs = nn.Sequential(*separable_convs)

        self.final = nn.Sequential(
            nn.AvgPool2d(dimensions[-1]),
            nn.Flatten(),
            nn.Linear(num_channels[-1], num_labels),
        )

    def forward(self, x):
        # (B, 3, H, W) -> (B, num_labels)
        # no softmax implemented
        x = self.initial(x)
        x = self.separable_convs(x)
        x = self.final(x)
        return x


def test():
    net = MobileNet(1000)
    data = torch.randn(5, 3, 512, 244)
    print(data.shape)

    net.eval()
    out = net(data)
    print(out.shape)


if __name__ == '__main__':
    test()
