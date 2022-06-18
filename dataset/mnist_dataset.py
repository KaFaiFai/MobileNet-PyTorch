import torchvision
from torchvision.transforms import transforms
from script.utils import gray2rgb


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, root: str, is_train: bool = True, transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(gray2rgb),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                transforms.Resize((28, 28))
            ])
        else:
            self.transform = transform
        super(MNISTDataset, self).__init__(root, train=is_train, transform=self.transform, download=True)

    @property
    def num_labels(self):
        return 10
