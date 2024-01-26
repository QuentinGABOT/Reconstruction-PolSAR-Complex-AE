import torch
import torch.nn as nn

# from .parts import Up, Down, Dense
from .parts import Down, Up, OutConv, DoubleConv


class AutoEncoder(nn.Module):
    def __init__(
        self,
        num_channels,
        num_layers,
        channels_ratio,
    ):
        super(AutoEncoder, self).__init__()
        self.n_channels = num_channels

        # Encoder with doubzling channels
        current_channels = channels_ratio
        self.encoder_layers = []
        self.encoder_layers.append(DoubleConv(self.n_channels, current_channels))
        for i in range(1, num_layers):
            out_channels = channels_ratio * 2**i
            self.encoder_layers.append(Down(current_channels, out_channels))
            current_channels = out_channels
        self.encoder = nn.Sequential(*self.encoder_layers)

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
        x = self.decoder(x)
        return x

    def use_checkpointing(self):
        self.encoder = torch.utils.checkpoint(self.encoder)
        self.decoder = torch.utils.checkpoint(self.decoder)
