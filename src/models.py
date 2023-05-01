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
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
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
    """
    Decoder class for VAE.
    """
    def __init__(self, output_dim, z_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, output_dim)
                                  )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.model(z)
        output = self.sigmoid(hidden)
        return output


class VAE(nn.Module):
    """
    VAE model.
    """
    def __init__(self, input_dim, z_dim=2, hidden_dim=128, use_cuda=True):
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
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            # decode latent z
            decoded = self.decoder(z)

            # score against actual data
            pyro.sample("obs",
                        dist.Bernoulli(decoded, validate_args=False).to_event(1),
                        obs=x.reshape(-1, self.input_dim)
                       )

    # define the guide, q(z|x)
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def forward(self, x):
        z_loc, z_scale = self.encoder(x)

        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()

        return self.decoder(z)
