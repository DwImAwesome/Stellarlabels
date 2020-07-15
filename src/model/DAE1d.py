import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.VAE1d import ConvEncoder1D, ConvDecoder1D

class DAE1d(nn.Module):
    def __init__(self, input_shape, latent_size):
        super().__init__()

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = ConvEncoder1D(input_shape, latent_size)
        self.decoder = ConvDecoder1D(latent_size, input_shape)

    def elbo(self, x, outputs, beta):
        x_hat, mu, log_std = outputs
        recon_loss = F.mse_loss(x, x_hat, reduction='none').view(x.shape[0], -1).sum(1).mean()

        return recon_loss, recon_loss, recon_loss

    def forward(self, x):
        x = x.squeeze(1)
        z, _ = self.encoder(x)
        x_recon = self.decoder(z)
        x_recon = x_recon.unsqueeze(1)
        return x_recon, z, _

    def reconstruct(self, x):
       x = x.squeeze(1)
       z, _ = self.encoder(x)
       x_recon = self.decoder(z)
       x_recon = x_recon.unsqueeze(1)
       return x_recon, z
