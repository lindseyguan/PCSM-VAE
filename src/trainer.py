"""
author: lguan (at) mit (dot) edu
ref: https://pyro.ai/examples/vae.html


Trains variational autoencoder for protein condensate 
and small molecule interaction data. Saves model.

"""

import os
import sys

import numpy as np
from tqdm import tqdm

from sklearn.metrics import jaccard_score

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import PyroLRScheduler, ExponentialLR, optim

sys.path.append('./')
sys.path.append('../')

from models import VAE
from datasets import InteractionDataset


DATA_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/data'
MODEL_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/models'


def train(train_path, val_path, model_dir, num_epochs=20, batch_size=32, use_cuda=True):
    pyro.clear_param_store()

    vae = VAE(200)
    optimizer = Adam
    scheduler = ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': 0.001}, 'gamma': 0.1})
    svi = SVI(vae.model, vae.guide, scheduler, loss=Trace_ELBO())

    train_ds = InteractionDataset(file_path=train_path)
    val_ds = InteractionDataset(file_path=val_path)
    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True
                             )
    val_loader = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            shuffle=True
                           )
    train_losses = []
    val_losses = []
    for i in (pbar := tqdm(range(num_epochs))):
        epoch_loss_train = 0
        for x in train_loader:
            if use_cuda and torch.cuda.is_available():
                x = x.cuda()
            epoch_loss_train += svi.step(x)

        epoch_loss_val = 0
        for x in val_loader:
            if use_cuda and torch.cuda.is_available():
                x = x.cuda()
            epoch_loss_val += svi.evaluate_loss(x)

        total_epoch_loss_train = epoch_loss_train / len(train_loader.dataset)
        total_epoch_loss_val = epoch_loss_val / len(val_loader.dataset)

        scheduler.step()

        train_losses.append(total_epoch_loss_train)
        val_losses.append(total_epoch_loss_val)

        pbar.set_description(f"train loss: {total_epoch_loss_train}, val_loss: {total_epoch_loss_val}")

    scheduler_state = scheduler.get_state()
    model_save = {'state_dict':vae.state_dict(),
                  'scheduler':scheduler_state,
                  'train_loss':train_losses,
                  'val_loss':val_losses
                 }

    torch.save(model_save, os.path.join(model_dir, 'model_checkpoint.pt'))
    return train_losses

def validate():
    pass

if __name__ == "__main__":
    train_loss = train(train_path=os.path.join(DATA_DIR, 'train_clean.csv'),
                       val_path=os.path.join(DATA_DIR, 'val_clean.csv'),
                       model_dir=MODEL_DIR
                      )
    print(train_loss)
