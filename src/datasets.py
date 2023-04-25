"""
author: lguan (at) mit (dot) edu
ref: https://pyro.ai/examples/vae.html


Implements InteractionDataset for protein condensate 
and small molecule interaction data. Used in VAE.

"""

import pandas as pd

from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path,
                                index_col=0
                               )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.transform:
            item = self.transform(item)
        return item.to_numpy().astype('float32')
