"""
To load the Stanford Dogs Dataset for image classification: http://vision.stanford.edu/aditya86/ImageNetDogs/
Usage:
1. Download
    - Images (757MB)
    - Lists, with train/test splits (0.5MB)
2. "tar -xf" both .tar files in the same root directory
3. Initiate with DogsDataset(root="/your/root/directory")
"""

from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io
from pathlib import Path
from PIL import Image


class DogsDataset(Dataset):

    def __init__(self, root: str, is_train: bool = True, transform=None):
        super().__init__()
        self.root_dir = Path(root)
        assert self.root_dir.is_dir()

        if is_train:
            self.mat_path = self.root_dir / "train_list.mat"
        else:
            self.mat_path = self.root_dir / "test_list.mat"
        self.mat = scipy.io.loadmat(str(self.mat_path))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True),
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        image_path = self.root_dir / "Images" / self.mat["file_list"][index][0][0]
        label = self.mat["labels"][index][0]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, int(label) - 1

    def __len__(self):
        return len(self.mat["file_list"])

    @classmethod
    def label2name(cls, label):
        return cls._classes[label]

    @property
    def num_class(self):
        return len(self._classes)

    _classes = (
        "Chihuaha",
        "Japanese Spaniel",
        "Maltese Dog",
        "Pekinese",
        "Shih-Tzu",
        "Blenheim Spaniel",
        "Papillon",
        "Toy Terrier",
        "Rhodesian Ridgeback",
        "Afghan Hound",
        "Basset Hound",
        "Beagle",
        "Bloodhound",
        "Bluetick",
        "Black-and-tan Coonhound",
        "Walker Hound",
        "English Foxhound",
        "Redbone",
        "Borzoi",
        "Irish Wolfhound",
        "Italian Greyhound",
        "Whippet",
        "Ibizian Hound",
        "Norwegian Elkhound",
        "Otterhound",
        "Saluki",
        "Scottish Deerhound",
        "Weimaraner",
        "Staffordshire Bullterrier",
        "American Staffordshire Terrier",
        "Bedlington Terrier",
        "Border Terrier",
        "Kerry Blue Terrier",
        "Irish Terrier",
        "Norfolk Terrier",
        "Norwich Terrier",
        "Yorkshire Terrier",
        "Wirehaired Fox Terrier",
        "Lakeland Terrier",
        "Sealyham Terrier",
        "Airedale",
        "Cairn",
        "Australian Terrier",
        "Dandi Dinmont",
        "Boston Bull",
        "Miniature Schnauzer",
        "Giant Schnauzer",
        "Standard Schnauzer",
        "Scotch Terrier",
        "Tibetan Terrier",
        "Silky Terrier",
        "Soft-coated Wheaten Terrier",
        "West Highland White Terrier",
        "Lhasa",
        "Flat-coated Retriever",
        "Curly-coater Retriever",
        "Golden Retriever",
        "Labrador Retriever",
        "Chesapeake Bay Retriever",
        "German Short-haired Pointer",
        "Vizsla",
        "English Setter",
        "Irish Setter",
        "Gordon Setter",
        "Brittany",
        "Clumber",
        "English Springer Spaniel",
        "Welsh Springer Spaniel",
        "Cocker Spaniel",
        "Sussex Spaniel",
        "Irish Water Spaniel",
        "Kuvasz",
        "Schipperke",
        "Groenendael",
        "Malinois",
        "Briard",
        "Kelpie",
        "Komondor",
        "Old English Sheepdog",
        "Shetland Sheepdog",
        "Collie",
        "Border Collie",
        "Bouvier des Flandres",
        "Rottweiler",
        "German Shepard",
        "Doberman",
        "Miniature Pinscher",
        "Greater Swiss Mountain Dog",
        "Bernese Mountain Dog",
        "Appenzeller",
        "EntleBucher",
        "Boxer",
        "Bull Mastiff",
        "Tibetan Mastiff",
        "French Bulldog",
        "Great Dane",
        "Saint Bernard",
        "Eskimo Dog",
        "Malamute",
        "Siberian Husky",
        "Affenpinscher",
        "Basenji",
        "Pug",
        "Leonberg",
        "Newfoundland",
        "Great Pyrenees",
        "Samoyed",
        "Pomeranian",
        "Chow",
        "Keeshond",
        "Brabancon Griffon",
        "Pembroke",
        "Cardigan",
        "Toy Poodle",
        "Miniature Poodle",
        "Standard Poodle",
        "Mexican Hairless",
        "Dingo",
        "Dhole",
        "African Hunting Dog",
    )
