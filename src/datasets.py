import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class InteractionDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path, 
        					    index_col=0
        					   )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.loc[idx]
        if self.transform:
            item = self.transform(item)
        return item
