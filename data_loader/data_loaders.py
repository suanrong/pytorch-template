from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader

class xxDataset(Dataset):
    def __init__(self, name):
        self.name = name
        self.data = None
        self.len = None

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return self.data.features[idx], self.data.labels[idx]

class MnistDataLoader(BaseDataLoader):
    """
    a sample MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
