import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from .models import VAE
from .datasets import InteractionDataset


DATA_DIR = '/Users/lindseyguan/Documents/zhang/data/vectors'


def train(train_path, model_dir, num_epochs=2, batch_size=64, use_cuda=True):
    vae = VAE()
    optimizer = Adam({"lr": 1e-3})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    transform = transforms.ToTensor()
    train_ds = InteractionDataset(train_path)
    train_loader = DataLoader(dataset=train_ds, 
    						  batch_size=batch_size, 
    						  shuffle=True
    						 )

    train_losses = []
    for i in range(num_epochs):
	    epoch_loss = 0
	    for x, _ in train_loader:
	        if use_cuda and torch.cuda.is_available():
	            x = x.cuda()
	        epoch_loss += svi.step(x)

	    normalizer_train = len(train_loader.dataset)
	    total_epoch_loss_train = epoch_loss / normalizer_train
	    train_losses.append(total_epoch_loss_train)

	torch.save(svi.model.state_dict(), os.path.join(model_dir, 'model.pt'))
	torch.save(svi.guide.state_dict(), os.path.join(model_dir, 'guide.pt'))
	np.save(np.array(train_losses), os.path.join(model_dir, 'train_loss.npy'))

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
	train(train_path='/Users/lindseyguan/Documents/zhang/PCSM-VAE/data/train_clean.csv',
		  model_dir='/Users/lindseyguan/Documents/zhang/PCSM-VAE/models'
		 )
