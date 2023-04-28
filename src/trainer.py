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


def evaluate(test_path, model_path, use_cuda=True):
    pyro.clear_param_store()
    if use_cuda and torch.cuda.is_available():
        model_checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))
    else:
        model_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    vae = VAE(200)
    vae.load_state_dict(model_checkpoint['state_dict'])
    vae.eval()

    test_ds = InteractionDataset(file_path=test_path)
    test_loader = DataLoader(dataset=test_ds,
                             batch_size=8
                            )

    test_losses = []
    test_jaccard = 0
    with torch.no_grad():
        for x in tqdm(test_loader):
            if use_cuda and torch.cuda.is_available():
                x = x.cuda()
            pred = vae.forward(x).numpy()
            print(x[0])
            print(pred[0])
            for threshold in np.linspace(0.2, 0.3, 100):
                thresholded_pred = [apply_threshold(p, threshold) for p in pred]
                print(threshold, jaccard_score(x[0].numpy(), thresholded_pred[0]))
            break

    return test_losses

def apply_threshold(arr, threshold):
    """Binarizes `arr` using the specified threshold.
    
    Args:
        arr: float or iterable of values between 0 and 1
        threshold: float between 0 and 1
    """
    try:
        arr_ = iter(arr)
    except TypeError:
        if arr >= 0 and arr <= 1:
            return 1 if arr >= threshold else 0
        else:
            raise ValueError('Input(s) must be between 0 and 1.')

    output = []
    for i in range(len(arr)):
        if arr[i] >= 0 and arr[i] <= 1:
            val = 1 if arr[i] >= threshold else 0
            output.append(val)
        else:
            raise ValueError('Input(s) must be between 0 and 1.')
    return output

if __name__ == "__main__":
    # print(train(train_path=os.path.join(DATA_DIR, 'train_clean.csv'),
    #             model_dir=MODEL_DIR
    #            )
    #      )
    evaluate(test_path=os.path.join(DATA_DIR, 'val_clean.csv'),
             model_path=os.path.join(MODEL_DIR, 'model_checkpoint.pt'))
