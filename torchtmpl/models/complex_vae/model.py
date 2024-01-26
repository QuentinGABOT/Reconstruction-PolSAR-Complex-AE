import torch
import torch.nn as nn
import torch.nn.functional as F

# from .parts import Up, Down, Dense, Reparametrize
from .parts import Down, Up, OutConv, DoubleConv, Dense


class VAE(nn.Module):
    def __init__(
        self,
        num_channels,
        num_layers,
        channels_ratio,
        latent_dim,
        input_size,
    ):
        super(VAE, self).__init__()
        self.n_channels = num_channels

        # Encoder with doubling channels
        current_channels = channels_ratio
        self.encoder_layers = []
        self.encoder_layers.append(DoubleConv(self.n_channels, current_channels))
        for i in range(1, num_layers):
            out_channels = channels_ratio * 2**i
            input_size //= 2
            self.encoder_layers.append(Down(current_channels, out_channels))
            current_channels = out_channels
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.dense = Dense(
            in_channels=current_channels,
            latent_dim=latent_dim,
            input_size=input_size,
        )
        """
        # Latent space layers
        self.fc_mu = nn.Linear(
            current_channels*self.input_size*self.input_size,
            latent_dim,
            dtype=torch.complex64,
        )
        self.fc_logvar = nn.Linear(
            current_channels*self.input_size*self.input_size,
            latent_dim,
            dtype=torch.complex64,
        )
        self.reparametrize = Reparametrize()
        self.decoder_input = nn.Linear(
            latent_dim,
            current_channels*self.input_size*self.input_size,
            dtype=torch.complex64,
        )
        """

        # Decoder with halving channels
        self.decoder_layers = []

        for i in range(num_layers - 2, -1, -1):
            out_channels = channels_ratio * 2**i
            self.decoder_layers.append(Up(current_channels, out_channels))
            current_channels = out_channels
        self.decoder_layers.append(OutConv(current_channels, num_channels))
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        """
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparametrize(mu, logvar)
        print(z.shape)
        input()
        
        #x = self.decoder_input(z)
        x = torch.unflatten(z, ())
        print(x.shape)
        input()
        #x = x.view(x.size(0), -1, 1, 1)  # Reshape for the decoder
        print(x.shape)
        input()
        """
        x, mu, logvar = self.dense(x)
        x = self.decoder(x)
        return x, mu, logvar

    def use_checkpointing(self):
        self.encoder = torch.utils.checkpoint(self.encoder)
        self.decoder = torch.utils.checkpoint(self.decoder)
