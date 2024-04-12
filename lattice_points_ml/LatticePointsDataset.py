import torch
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pandas as pd
import os


class LatticePointsDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.annot_filename = annotations_file
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path: str = self.img_labels.iloc[idx, 0]
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


