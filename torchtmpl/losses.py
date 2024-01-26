import torch
import torch.nn as nn


class ComplexMeanSquareError(nn.Module):
    def __init__(self):
        super(ComplexMeanSquareError, self).__init__()

    def forward(self, y_true, y_pred):
        # Ensure y_pred and y_true are torch tensors
        y_pred = torch.as_tensor(y_pred)
        y_true = torch.as_tensor(y_true)

        # Handle complex pred but real true
        if torch.is_complex(y_pred) and not torch.is_complex(y_true):
            y_true = torch.complex(y_true, torch.zeros_like(y_true))

        # Ensure y_true is the same dtype as y_pred
        y_true = y_true.type_as(y_pred)

        # Calculate Mean Square Error
        mse = torch.mean(torch.square(torch.abs(y_true - y_pred)))

        return mse


class ComplexVAELoss(nn.Module):
    def __init__(self):
        """
        Initializes the VAE Loss module.

        """
        super(ComplexVAELoss, self).__init__()

    def forward(self, x, recon_x, mu, logvar):
        """
        Computes the VAE loss.

        Parameters:
        recon_x: Reconstructed data.
        x: Original input data.
        mu: Mean from the latent space.
        logvar: Log variance from the latent space.

        Returns:
        torch.Tensor: Computed VAE loss.
        """
        # Reconstruction Loss-
        MSELoss = ComplexMeanSquareError()
        recon_loss = MSELoss(y_true=x, y_pred=recon_x)

        # KL Divergence
        kl_divergence = torch.abs(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()))
        return recon_loss + kl_divergence


# Example usage
# loss_fn = ComplexMeanSquareError()
# loss = loss_fn(y_true, y_pred)
