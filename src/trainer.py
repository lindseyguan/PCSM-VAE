"""
author: lguan (at) mit (dot) edu
ref: https://pyro.ai/examples/vae.html


Trains variational autoencoder for protein condensate 
and small molecule interaction data. Saves model.

"""

import os
import sys

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

sys.path.append('./')
sys.path.append('../')

from models import VAE
from datasets import InteractionDataset


DATA_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/data'
MODEL_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/models'


def train(train_path, model_dir, num_epochs=5, batch_size=64, use_cuda=True):
    pyro.clear_param_store()

    vae = VAE(200)
    optimizer = Adam({"lr": 1e-3})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_ds = InteractionDataset(file_path=train_path)
    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True
                             )

    train_losses = []
    for i in tqdm(range(num_epochs)):
        epoch_loss = 0
        for x in tqdm(train_loader):
            if use_cuda and torch.cuda.is_available():
                x = x.cuda()
            loss = svi.step(x)
            epoch_loss += loss

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_losses.append(total_epoch_loss_train)

    model_save = {'state_dict':vae.state_dict(),
                  'optimizer':optimizer.get_state(),
                  'train_loss':train_losses
                 }
    torch.save(model_save, os.path.join(model_dir, 'model_checkpoint.pt'))
    return train_losses


def evaluate(test_path, model_dir, use_cuda=True):
    # TODO: fix this
    test_loss = 0
    for x, _ in test_loader:
        if use_cuda and torch.cuda.is_available():
            x = x.cuda()
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


if __name__ == "__main__":
    print(train(train_path=os.path.join(DATA_DIR, 'train_clean.csv'),
                model_dir=MODEL_DIR
               )
         )
