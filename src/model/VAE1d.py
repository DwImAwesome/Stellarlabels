import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ConvDecoder1D(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        #self.base_size = (128, output_shape[1] // 8, output_shape[2] // 8)
        self.base_size = (128, output_shape[1] // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, output_shape[0], 3, padding=1),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], *self.base_size)
        out = self.deconvs(out)
        return out


class ConvEncoder1D(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv1d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            #ResidualStack1D(64),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
        )
        #conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        conv_out_dim = input_shape[1] // 8 * 256
        self.fc = nn.Linear(conv_out_dim, 2 * latent_dim)

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        mu, log_std = self.fc(out).chunk(2, dim=1)
        return mu, log_std


class ConvVAE1D(nn.Module):
    def __init__(self, input_shape, latent_size):
        super().__init__()
        #assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = ConvEncoder1D(input_shape, latent_size)
        self.decoder = ConvDecoder1D(latent_size, input_shape)

    def elbo(self, x, outputs, beta):
        x_hat, mu, log_std = outputs
        recon_loss = F.mse_loss(x, x_hat, reduction='none').view(x.shape[0], -1).sum(1).mean()
        kl_loss = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
        kl_loss = kl_loss.sum(1).mean()

        return recon_loss + beta*kl_loss, recon_loss, kl_loss

    def forward(self, x):
        x = x.squeeze(1)
        mu, log_std = self.encoder(x)
        z = torch.randn_like(mu) * log_std.exp() + mu
        x_recon = self.decoder(z)
        x_recon = x_recon.unsqueeze(1)
        return x_recon, mu, log_std

    def reconstruct(self, x):
       x = x.squeeze(1)
       z, log_std = self.encoder(x)
       #z = torch.randn_like(z) * log_std.exp() + z
       x_recon = self.decoder(z)
       x_recon = x_recon.unsqueeze(1)
       return x_recon, z
