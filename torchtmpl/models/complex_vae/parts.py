""" Parts of the VAE model """

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn
from math import prod


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self, in_channels, out_channels, activation, stride=1, mid_channels=None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="replicate",
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(mid_channels),
            activation,
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                padding_mode="replicate",
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(
                in_channels,
                out_channels,
                activation,
                stride=2,
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.up = c_nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels, out_channels, activation)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.complex64),
        )

    def forward(self, x):
        return self.conv(x)


class Dense(nn.Module):
    def __init__(self, in_channels, latent_dim, input_size):
        super(Dense, self).__init__()
        # Latent space layers
        linear = in_channels * input_size * input_size
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_mu = nn.Linear(
            in_features=linear, out_features=latent_dim, dtype=torch.complex64
        )
        self.fc_covar = nn.Sequential(
            nn.Linear(
                in_features=linear,
                out_features=latent_dim,
                dtype=torch.complex64,
            ),
            c_nn.activation.Mod(),
        )
        self.fc_pseudo_covar = nn.Linear(
            in_features=linear, out_features=latent_dim, dtype=torch.complex64
        )
        self.unflatten = nn.Sequential(
            nn.Linear(latent_dim, linear, dtype=torch.complex64),
            nn.Unflatten(dim=1, unflattened_size=(in_channels, input_size, input_size)),
        )

    def reparametrize(self, mu, sigma, delta):

        z = torch.randn_like(sigma, dtype=torch.complex64)  # z ~ CN(0,I,O)
        xy = torch.stack((z.real, z.imag), dim=2)
        Cx = torch.stack((sigma + delta.real, delta.imag), dim=2)
        Cy = torch.stack((delta.imag, sigma - delta.real), dim=2)
        Cxy = torch.stack((Cx, Cy), dim=2)

        L = torch.linalg.cholesky(Cxy)
        xy = xy.unsqueeze(-1)
        z_tilde = torch.matmul(L, xy)
        z_tilde = z_tilde.squeeze(-1)

        x_tilde = z_tilde[:, :, 0] + mu.real
        y_tilde = z_tilde[:, :, 1] + mu.imag
        return x_tilde + 1j * y_tilde

    def forward(self, x):
        x = self.flatten(x)
        mu = self.fc_mu(x)
        sigma = self.fc_covar(x)
        delta = self.fc_pseudo_covar(x)
        delta = torch.where(
            torch.abs(delta) > 0.99 * sigma,
            0.99 * sigma * torch.exp(1j * torch.angle(delta)),
            delta,
        )
        z = self.reparametrize(mu, sigma, delta)
        x = self.unflatten(z)
        return x, mu, sigma, delta
