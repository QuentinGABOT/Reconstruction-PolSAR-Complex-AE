import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import logging
import random

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms

from torchcvnn.datasets.polsf import PolSFAlos2Dataset


def equalize(image, p2=None, p98=None):
    """
    Automatically Ajust contrast of the SAR image
    Input: intensity or amplitude  in dB scale
    """
    img = np.log10(np.abs(image) + 1e-15)
    if not p2:
        p2, p98 = np.percentile(img, (1, 99))
    img_resc = exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))

    return (20 * img, img_resc)


def show_images(samples, generated, image_path):
    num_samples = len(samples)
    fig, axes = plt.subplots(
        num_samples, 2, figsize=(10, 5 * num_samples)
    )  # Adjust layout for multiple samples

    for i in range(num_samples):
        img_dataset, img_gen = samples[i], generated[i]
        # Equalize and transpose if needed
        _, img_dataset = equalize(img_dataset.transpose(1, 2, 0))
        _, img_gen = equalize(img_gen.transpose(1, 2, 0))

        img_dataset = np.round(img_dataset * 255).astype(np.uint8)
        img_gen = np.round(img_gen * 255).astype(np.uint8)

        # Plot dataset image
        axes[i][0].imshow(img_dataset, vmax=1e-1, origin="lower")
        axes[i][0].set_title(f"Image dataset {i+1}")

        # Plot generated image
        axes[i][1].imshow(img_gen, vmax=1e-1, origin="lower")
        axes[i][1].set_title(f"Image generated {i+1}")

        # Remove axis labels
        for ax in axes[i]:
            ax.axis("off")

    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close()


"""
def show_image(img_dataset, img_gen, logdir, epoch):
    _, img_dataset = equalize(img_dataset.transpose(1, 2, 0))
    _, img_gen = equalize(img_gen.transpose(1, 2, 0))

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first image on the left subplot
    axes[0].imshow(img_dataset, vmax=1e-1, origin="lower")
    axes[0].set_title("Image dataset")

    # Plot the second image on the right subplot
    axes[1].imshow(img_gen, vmax=1e-1, origin="lower")
    axes[1].set_title("Image generated")

    # Remove axis labels and display the images
    for ax in axes:
        ax.axis("off")

    plt.savefig(logdir / f" output_{epoch}.png", bbox_inches="tight", pad_inches=0)

    plt.close()
"""


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose([transforms.ToTensor()])

    base_dataset = PolSFAlos2Dataset(root=data_config["trainpath"], download=False)

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    # indices = list(range(len(base_dataset)))
    indices = list(range(100))
    random.shuffle(indices)
    # num_valid = int(valid_ratio * len(base_dataset))
    num_valid = int(valid_ratio * len(indices))
    # train_indices = indices[num_valid:]
    # valid_indices = indices[:num_valid]
    # train_indices = [78, 98] * 500
    # valid_indices = [78, 98] * 500
    train_indices = [98] * 500
    valid_indices = [98] * 500

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    input_size = tuple(
        base_dataset[0].shape
    )  # size cause the dataset is made of tensor and not PIL nor nparray

    return train_loader, valid_loader, input_size
