"""
Implements VAE for protein condensate + small molecule simulation data.

Based on Pyro's VAE code: https://pyro.ai/examples/vae.html.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
						           nn.ReLU(),
						           nn.Linear(hidden_dim, hidden_dim),
						           nn.ReLU()
        						  )

        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_cov = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
    	# Outputs latent Gaussians
        hidden = self.model(x)
        z_mean = self.fc_mean(hidden) # means
        z_scale = torch.exp(self.fc_cov(hidden)) # square root covariances
        return z_mean, z_scale


class Decoder(nn.Module):
    def __init__(self, output_dim, z_dim, hidden_dim):
        super().__init__()
	    self.model = nn.Sequential(nn.Linear(z_dim, hidden_dim),
						           nn.ReLU(),
						           nn.Linear(hidden_dim, hidden_dim),
						           nn.ReLU()
        						  )

        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_cov = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
    	# Outputs Gaussians
        hidden = self.model(z)
        out_mean = self.fc_mean(hidden) # means
        out_scale = torch.exp(self.fc_cov(hidden)) # square root covariances
        return out_mean, out_scale


 class VAE(nn.Module):
    def __init__(self, input_dim, z_dim=2, hidden_dim=48, use_cuda=True):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, 
        					   z_dim=z_dim, 
        					   hidden_dim=hidden_dim)

        self.decoder = Decoder(output_dim=input_dim, 
        					   z_dim=z_dim, 
        					   hidden_dim=hidden_dim)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model, p(x|z)p(z)
    def model(self, x):
        # register decoder with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            # decode the latent code z
            decoded_mean, decoded_scale = self.decoder(z)

            # score against actual data
            pyro.sample("obs", 
            			dist.Normal(decoded_mean, decoded_scale).to_event(1), 
            			obs=x.reshape(-1, self.input_dim)
            		   )

    # define the guide, q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing data
    def reconstruct_data(self, x):
        z_loc, z_scale = self.encoder(x)

        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()

        decoded_mean, decoded_scale = self.decoder(z)
        return decoded_mean, decoded_scale

