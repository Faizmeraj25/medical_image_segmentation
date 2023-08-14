from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import datasets, transforms
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random

transformer1 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4602, 0.4495, 0.3800], std=[0.2040, 0.1984, 0.1921])
    ])
train_Dataset = torchvision.datasets.ImageFolder(train_path, transform=transformer1)
dataset = datasets.ImageFolder('/Dataset/', transform=transform)
print(dataset)