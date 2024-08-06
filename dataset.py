from torch.utils.data import Dataset, WeightedRandomSampler, Subset
from PIL import Image
import numpy as np
import os
import pandas as pd
import torch

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