from torch.utils.data import Dataset, WeightedRandomSampler, Subset, DataLoader
from PIL import Image
import numpy as np
import os
import pandas as pd
import torch

import medmnist
from medmnist import INFO

from torchvision.transforms.functional import InterpolationMode


# Define the custom dataset
class ISICDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.classes = sorted(set(labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_full_isic(augments, norm_only):
    metadata = pd.read_csv('data/metadata.csv')
    labels = metadata['malignant'].values.astype(int)
    files = [f"data/ISIC_2024_Training_Input/{f}" for f in os.listdir('data/ISIC_2024_Training_Input') if f.endswith('.jpg')]

    dataset_train = ISICDataset(files, labels, transform=augments)
    dataset_knn = ISICDataset(files, labels, transform=norm_only)

    return dataset_train, dataset_knn


class MedMNIST:
    def __init__(self, batch_size, name, train_augs, norm_augs, seed=42):
        self.batch_size = batch_size
        self.seed = seed
        self.name = name
        self.as_rgb = True
        self.train_augs = train_augs
        self.norm_augs = norm_augs

    def get_loaders(self):
        info = INFO[self.name]
        DataClass = getattr(medmnist, info["python_class"])

        train_dataset = DataClass(
            split="train", transform=self.train_augs, download=True, as_rgb=self.as_rgb, size=224
        )
        val_dataset = DataClass(
            split="val", transform=self.norm_augs, download=True, as_rgb=self.as_rgb, size=224
        )
        test_dataset = DataClass(
            split="test", transform=self.norm_augs, download=True, as_rgb=self.as_rgb, size=224
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

        return train_loader, val_loader, test_loader