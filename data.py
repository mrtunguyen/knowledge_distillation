from pathlib import Path
from typing import Union
import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as plt

class DataModule(plt.LightningDataModule):
    def __init__(self,
                 data_name: str = "CIFAR10", 
                 data_path: Union[str, Path] = "./data", 
                 batch_size: int = 64,
                 download: bool = False):
        super().__init__()

        self.data_path = data_path
        self.data_name = data_name
        self.batch_size = batch_size
        self.download = download

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)), 
                transforms.ToTensor(), 
                self.normalize_transform
            ]
        )
    
    def prepare_dataset(self):
        self.train_dataset = torchvision.datasets.__dict__[self.data_name](root = self.data_path,
                                                                           train = True,
                                                                           download = self.download,
                                                                           transform = self.train_transform)
        self.val_dataset = torchvision.datasets.__dict__[self.data_name](root = self.data_path,
                                                                         train = False,
                                                                         download = self.download,
                                                                         transform = self.valid_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )

if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_dataset()
    print(next(iter(data_model.train_dataloader())))