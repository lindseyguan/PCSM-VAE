"""
author: lguan (at) mit (dot) edu
ref: https://pyro.ai/examples/vae.html


Trains variational autoencoder for protein condensate 
and small molecule interaction data. Saves model.

"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

from sklearn.metrics import jaccard_score

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import PyroLRScheduler, ExponentialLR, optim, Adam

sys.path.append('./')
sys.path.append('../')

from models import VAE
from datasets import InteractionDataset


DATA_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/data'
MODEL_DIR = '/home/gridsan/lguan/zhang/PCSM-VAE/models'


def train(train_path, val_path, model_dir, 
          input_dim=239, 
          z_dim=16, 
          hidden_dim=64, 
          num_epochs=50, 
          batch_size=128, 
          use_cuda=True,
          name=''):
    pyro.clear_param_store()

    vae = VAE(input_dim=input_dim, 
              z_dim=z_dim, 
              hidden_dim=hidden_dim, 
              use_cuda=True
              )
    adam_args = {"lr": 1e-3}
    optimizer = Adam(adam_args)

    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

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
        for x, label in train_loader:
            if use_cuda and torch.cuda.is_available():
                x = x.cuda()
            epoch_loss_train += svi.step(x)

        epoch_loss_val = 0
        for x, label in val_loader:
            if use_cuda and torch.cuda.is_available():
                x = x.cuda()
            epoch_loss_val += svi.evaluate_loss(x)

        total_epoch_loss_train = epoch_loss_train / len(train_loader.dataset)
        total_epoch_loss_val = epoch_loss_val / len(val_loader.dataset)

        train_losses.append(total_epoch_loss_train)
        val_losses.append(total_epoch_loss_val)

        pbar.set_description(f"train loss: {total_epoch_loss_train}, val_loss: {total_epoch_loss_val}")

    model_save = {'z_dim':z_dim,
                  'input_dim':input_dim,
                  'hidden_dim':hidden_dim,
                  'state_dict':vae.state_dict(),
                  'optimizer':optimizer.get_state(),
                  'train_loss':train_losses,
                  'val_loss':val_losses
                 }

    if name != '':
        model_name = f'z{z_dim}_hidden{hidden_dim}_epochs{num_epochs}_bs{batch_size}_{name}.pt'
    else:
        model_name = f'z{z_dim}_hidden{hidden_dim}_epochs{num_epochs}_bs{batch_size}.pt'
        
    torch.save(model_save, os.path.join(model_dir, model_name))
    return train_losses, val_losses

def validate():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCSM VAE training parameters')

    parser.add_argument('--train_data',
                        type=str, 
                        default='train_clean.csv', 
                        help='name of training data file')
    parser.add_argument('--val_data',
                        type=str, 
                        default='val_clean.csv', 
                        help='name of validation data file')
    parser.add_argument('--input_dim',
                        type=int, 
                        default=239, 
                        help='input dimension')
    parser.add_argument('--z_dim',
                        type=int, 
                        default=16, 
                        help='dimension of z latent')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=64,
                        help='hidden dimension'
                       )
    parser.add_argument('--use_cuda',
                        default=False,
                        action='store_true',
                        help='whether to use cuda for GPU'
                       )
    parser.add_argument('--num_epochs',
                        default=50,
                        type=int,
                        help='number of epochs'
                       )
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='batch size'
                       )
    parser.add_argument('--name',
                        default='',
                        type=str,
                        help='additional name to distinguish this model -- will be added to output path'
                       )


    args = vars(parser.parse_args())

    train_loss, val_loss = train(train_path=os.path.join(DATA_DIR, args['train_data']),
                                 val_path=os.path.join(DATA_DIR, args['val_data']),
                                 model_dir=MODEL_DIR,
                                 input_dim=args['input_dim'],
                                 z_dim=args['z_dim'],
                                 hidden_dim=args['hidden_dim'],
                                 num_epochs=args['num_epochs'],
                                 batch_size=args['batch_size'],
                                )

    print('train loss:', train_loss)
    print('val loss:', val_loss)
