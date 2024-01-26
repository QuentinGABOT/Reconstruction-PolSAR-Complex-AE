import torch
import torch.nn as nn

from .parts import Down, Up, OutConv, DoubleConv


class UNet(nn.Module):
    def __init__(
        self,
        num_channels,
        num_layers,
        channels_ratio,
    ):
        super(UNet, self).__init__()
        self.n_channels = num_channels

        # Encoder with doubling channels
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
        skip_connections = []
        for enc in self.encoder_layers:
            x = enc(x)
            skip_connections.append(x)

        # Reverse the skip connections for use in the decoder
        skip_connections = skip_connections[::-1]

        for dec in self.decoder_layers:
            if len(skip_connections) > 1:
                x = dec(skip_connections[0], skip_connections[1])
                skip_connections = skip_connections[2:]
                skip_connections.insert(0, x)
            else:
                x = dec(skip_connections[0])

        return x

    def use_checkpointing(self):
        self.encoder = torch.utils.checkpoint(self.encoder)
        self.decoder = torch.utils.checkpoint(self.decoder)
