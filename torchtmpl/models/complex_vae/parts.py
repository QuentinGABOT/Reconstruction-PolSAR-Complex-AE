""" Parts of the VAE model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn
from math import prod


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(mid_channels),
            c_nn.zLeakyReLU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(out_channels),
            c_nn.CCELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            c_nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = c_nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.complex64),
            c_nn.modReLU(),
        )

    def forward(self, x):
        return self.conv(x)


"""
class Reparametrize(nn.Module):
    
    #Reparameterization trick for VAE, sampling from N(mu, logvar)
    

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
"""


class Dense(nn.Module):
    def __init__(self, in_channels, latent_dim, input_size):
        super(Dense, self).__init__()
        # Latent space layers
        linear = in_channels * input_size * input_size
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_mu = nn.Linear(
            in_features=linear, out_features=latent_dim, dtype=torch.complex64
        )
        self.fc_logvar = nn.Linear(
            in_features=linear, out_features=latent_dim, dtype=torch.complex64
        )
        self.unflatten = nn.Sequential(
            nn.Linear(latent_dim, linear, dtype=torch.complex64),
            nn.Unflatten(dim=1, unflattened_size=(in_channels, input_size, input_size)),
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparametrize(mu, logvar)
        x = self.unflatten(z)
        return x, mu, logvar
