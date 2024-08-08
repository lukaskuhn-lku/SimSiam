from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop, RandomRotation, ColorJitter, RandomGrayscale, RandomApply
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm

import torch
import torch.nn as nn

from dataset import load_full_isic, MedMNIST

import math

import wandb

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def compute_knn(backbone, data_loader_train, data_loader_val):
    device = next(backbone.parameters()).device

    data_loaders = {
        "train": data_loader_train,
        "val": data_loader_val,
    }

    lists = {
        "X_train": [],
        "y_train": [],
        "X_val": [],
        "y_val": [],
    }

    for name, data_loader in data_loaders.items():
        for imgs, y in data_loader:
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k,l in lists.items()}
    
    estimator = KNeighborsClassifier(8)
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)

    return acc, y_val_pred

#### CONFIGURATION ####
epochs = 100
num_workers = 8
batch_size = 256
pin_memory = True
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# if checkpoints folder does not exist, create it
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

class SimSiamAugmentations:
    def __init__(self, global_crops_scale=(0.2, 1.0), size=224):
        self.global_crops_scale = global_crops_scale
        self.image_size = size

        self.augmentations = Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(self.image_size, scale=global_crops_scale),
            RandomRotation(10),
            RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
            RandomGrayscale(p=0.2),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.augmentations(x), self.augmentations(x)

class SimSiamWrapper(nn.Module):
    def __init__(self, base_encoder, dim, pred_dim):
        super(SimSiamWrapper, self).__init__()

        self.encoder = base_encoder 
        self.encoder.head = nn.Identity() # if we remove the head we should be able to use this as is

        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim))
        
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach() # detach the z's as a stop-gradient

def train():
    wandb.init(
        project="SimSiam MedMNISt",
        tags=["simsiam", "medmnist", "vit"],
    )

    norm_only = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader, val_loader, _ = MedMNIST(batch_size, "dermamnist", SimSiamAugmentations(), norm_only).get_loaders()

    base_encoder, dim = timm.create_model("deit_tiny_patch16_224", pretrained=False), 192
    model = SimSiamWrapper(base_encoder, dim, 512).to(device)
    model.train()

    criterion = nn.CosineSimilarity(dim=1).to(device)
    lr = 0.05 * batch_size / 256
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for e in range(epochs):
        with tqdm(train_loader, unit='batch') as t:
            t.set_description(f"Epoch {e+1}")
            for images, _ in t:
                x1, x2 = images[0].to(device), images[1].to(device)

                p1, p2, z1, z2 = model(x1, x2)

                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                
                wandb.log({"loss": loss.item()})
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=loss.item())
            
            adjust_learning_rate(optimizer, lr, e, epochs)
            compute_knn(model.encoder, train_loader, val_loader)
            

        torch.save(model.encoder.state_dict(), f"checkpoints/model_{e}.pt")

if __name__ == "__main__":
    train()