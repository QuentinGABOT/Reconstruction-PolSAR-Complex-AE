# Local imports

from .complex_autoencoder.model import AutoEncoder
from .complex_unet.model import UNet
from .complex_vae.model import VAE

"""
def build_model(cfg):
    print(eval(f"{cfg['class']}"))
    input()
    print(eval(f"{cfg['model']['class'](**cfg['model']['params'])}"))
    input()
    return eval(f"{cfg['class']}{**cfg['params']}")
"""


def build_model(cfg):
    num_channels = cfg["data"]["num_channels"]
    num_layers = cfg["model"]["num_layers"]
    channels_ratio = cfg["model"]["channels_ratio"]
    latent_dim = cfg["model"]["latent_dim"]
    img_size = cfg["data"]["img_size"]
    model = cfg["model"]["class"]

    return eval(
        f"{model}(num_channels, num_layers, channels_ratio, latent_dim, img_size)"
    )
