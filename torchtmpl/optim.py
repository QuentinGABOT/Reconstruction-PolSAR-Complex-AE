# External imports
import torch
import torch.nn as nn
from torchcvnn.nn.modules.loss import ComplexMSELoss
from torchtmpl.losses import ComplexVAELoss, ComplexVAEPhaseLoss


def get_loss(lossname):
    return eval(f"{lossname}()")
    # return eval(f"nn.{lossname}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim
