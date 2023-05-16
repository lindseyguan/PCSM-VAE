"""
author: lguan (at) mit (dot) edu
ref: https://pyro.ai/examples/vae.html


Implements variational autoencoder for protein condensate 
and small molecule interaction data.

"""

import torch
from torch import nn

import pyro
import pyro.distributions as dist


class Encoder(nn.Module):
    """
    Encoder class for VAE.
    """
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim

        self.model_hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                         )
        self.model_mean = nn.Linear(hidden_dim, z_dim)
        self.model_cov = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        hidden = self.model_hidden(x)
        z_loc = self.model_mean(hidden)
        z_scale = torch.exp(self.model_cov(hidden))
        return z_loc, z_scale



class Decoder(nn.Module):
    """
    Decoder class for VAE.
    """
    def __init__(self, output_dim, z_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, output_dim),
                                  )


    def forward(self, z):
        output = self.model(z)
        recon = torch.sigmoid(output)
        return recon


class VAE(nn.Module):
    """
    VAE model.
    """
    def __init__(self, input_dim, z_dim=2, hidden_dim=16, use_cuda=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.z_dim = z_dim

        self.encoder = Encoder(input_dim=self.input_dim,
                               z_dim=self.z_dim,
                               hidden_dim=self.hidden_dim
                              )

        self.decoder = Decoder(output_dim=self.input_dim,
                               z_dim=self.z_dim,
                               hidden_dim=self.hidden_dim
                              )

        if use_cuda and torch.cuda.is_available():
            self.cuda()

    # define the model, p(x|z)p(z)
    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            recon = self.decoder.forward(z)
            pyro.sample(
                "obs",
                dist.Bernoulli(recon, validate_args=False).to_event(1),
                obs=x.reshape(-1, self.input_dim),
            )
            return recon

    # define the guide, q(z|x)
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct(self, x, with_sampling=False):
        z_loc, z_scale = self.encoder(x)
        if with_sampling:
            z = dist.Normal(z_loc, z_scale).sample()
        else:
            z = z_loc
        recon = self.decoder(z)
        return recon

    def encode(self, x, with_sampling=False):
        z_loc, z_scale = self.encoder(x)
        if with_sampling:
            z = dist.Normal(z_loc, z_scale).sample()
        else:
            z = z_loc
        return z
