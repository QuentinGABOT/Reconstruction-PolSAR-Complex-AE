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

'''
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
'''

from numpy.fft import fftshift


def equalize(image, p2=None, p98=None):
    """
    Automatically adjust contrast of the SAR image
    Input: intensity or amplitude in dB scale
    """
    img = np.log10(np.abs(image) + 1e-15)
    if not p2:
        p2, p98 = np.percentile(img, (1, 99))
    img_resc = np.round(
        exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1)) * 255
    ).astype(np.uint8)

    return img_resc


def plot_phase(image):
    """
    Plot the phase of a PolSAR image and normalize it to [0, 255].
    """
    phase_image = np.angle(image)  # Phase in [-π, π)
    # Normalize phase to [0, 1]
    normalized_phase = (phase_image + np.pi) / (2 * np.pi)
    # Scale to [0, 255] and convert to integer
    scaled_phase = np.round(normalized_phase * 255).astype(np.uint8)
    return scaled_phase


def plot_fourier_transform_amplitude_phase(image):
    amplitude_ft_images = []
    phase_ft_vectors = []

    for channel in range(image.shape[0]):
        fft_img = np.fft.fftshift(np.fft.fft2(image[channel, :, :]))
        amplitude = np.abs(fft_img)
        phase = np.angle(fft_img)

        # Compute magnitude spectrum and use log scale for visibility
        amplitude_ft_magnitude = np.log(np.abs(amplitude) + 1)
        phase_vectors = (np.cos(phase), np.sin(phase))

        amplitude_ft_images.append(amplitude_ft_magnitude)
        phase_ft_vectors.append(phase_vectors)

    return amplitude_ft_images, phase_ft_vectors


"""
def show_images(samples, generated, image_path, last):
    if last:
        num_samples = len(samples)
        num_channels = samples[0].shape[
            0
        ]  # Assuming 3 channels for PolSAR images: HH, HV(=VH), VV
        # Correct number of columns is indeed 2 (amplitude and phase) + 4 * num_channels (FT amplitude and phase for each channel)
        ncols = 4 + 4 * num_channels
        fig, axes = plt.subplots(
            nrows=num_samples,
            ncols=ncols,  # Updated to correct number of columns
            figsize=(5 * ncols, 5 * num_samples),
        )

        if num_samples == 1:
            axes = np.expand_dims(axes, 0)  # Ensure axes is a 2D array for consistency

        for i in range(num_samples):
            img_dataset, img_gen = samples[i], generated[i]
            img_dataset_trans = img_dataset.transpose(1, 2, 0)
            img_gen_trans = img_gen.transpose(1, 2, 0)

            # Plot amplitude and phase images
            axes[i][0].imshow(equalize(img_dataset_trans), vmax=1e-1, origin="lower")
            axes[i][0].set_title(f"Amplitude dataset {i+1}")
            axes[i][1].imshow(equalize(img_gen_trans), vmax=1e-1, origin="lower")
            axes[i][1].set_title(f"Amplitude generated {i+1}")
            axes[i][2].imshow(plot_phase(img_dataset_trans), cmap="hsv")
            axes[i][2].set_title(f"Phase dataset {i+1}")
            axes[i][3].imshow(plot_phase(img_gen_trans), cmap="hsv")
            axes[i][3].set_title(f"Phase generated {i+1}")

            # Compute and plot Fourier transforms for amplitude and phase for each channel
            dataset_amplitude_ft, dataset_phase_vectors = (
                plot_fourier_transform_amplitude_phase(img_dataset)
            )
            generated_amplitude_ft, generated_phase_vectors = (
                plot_fourier_transform_amplitude_phase(img_gen)
            )

            ax_index_phase = 2
            for ch in range(num_channels):
                # Correcting the indexing for Fourier transform plots
                ax_index_amplitude = ax_index_phase + 2  # Index for amplitude FT plots
                ax_index_phase = (
                    ax_index_amplitude + 2
                )  # Index for phase FT plots, immediately following amplitude plots

                axes[i][ax_index_amplitude].imshow(
                    dataset_amplitude_ft[ch], cmap="gray"
                )
                axes[i][ax_index_amplitude].set_title(f"FT Amp Dataset Ch{ch+1}")
                axes[i][ax_index_amplitude + 1].imshow(
                    generated_amplitude_ft[ch], cmap="gray"
                )
                axes[i][ax_index_amplitude + 1].set_title(f"FT Amp Generated Ch{ch+1}")

                X, Y = np.meshgrid(
                    np.arange(samples[0][ch, :, :].shape[1]),
                    np.arange(samples[0][ch, :, :].shape[0]),
                )
                axes[i][ax_index_phase].quiver(
                    X,
                    Y,
                    dataset_phase_vectors[ch][0],
                    dataset_phase_vectors[ch][1],
                    scale=60,
                )
                axes[i][ax_index_phase].set_title(f"FT Phase Dataset Ch{ch+1}")

                axes[i][ax_index_phase + 1].quiver(
                    X,
                    Y,
                    generated_phase_vectors[ch][0],
                    generated_phase_vectors[ch][1],
                    scale=60,
                )
                axes[i][ax_index_phase + 1].set_title(f"FT Phase Generated Ch{ch+1}")

            # Remove axis labels
            for ax in axes.flatten():
                ax.axis("off")

        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    else:
        num_samples = len(samples)
        fig, axes = plt.subplots(
            num_samples, 2, figsize=(10, 5 * num_samples)
        )  # Adjust layout for multiple samples

        for i in range(num_samples):
            img_dataset, img_gen = samples[i], generated[i]
            img_dataset_trans = img_dataset.transpose(1, 2, 0)
            img_gen_trans = img_gen.transpose(1, 2, 0)

            # Plot amplitude and phase images
            axes[i][0].imshow(equalize(img_dataset_trans), vmax=1e-1, origin="lower")
            axes[i][0].set_title(f"Amplitude dataset {i+1}")
            axes[i][1].imshow(equalize(img_gen_trans), vmax=1e-1, origin="lower")
            axes[i][1].set_title(f"Amplitude generated {i+1}")

            # Remove axis labels
            for ax in axes[i]:
                ax.axis("off")

        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
        plt.close()
"""


def show_images(samples, generated, image_path, last):
    num_samples = len(samples)
    num_channels = samples[0].shape[
        0
    ]  # Assuming 3 channels for PolSAR images: HH, HV(=VH), VV

    if last:
        ncols = 4 + 4 * num_channels  # Correct number of columns for detailed view
    else:
        ncols = 2  # Only amplitude images when not 'last'

    fig, axes = plt.subplots(
        nrows=num_samples, ncols=ncols, figsize=(5 * ncols, 5 * num_samples)
    )
    axes = np.atleast_2d(axes)  # Ensure axes is a 2D array for consistency

    for i in range(num_samples):
        img_dataset, img_gen = samples[i], generated[i]
        img_dataset_trans = img_dataset.transpose(1, 2, 0)
        img_gen_trans = img_gen.transpose(1, 2, 0)

        # Plot amplitude images
        axes[i][0].imshow(equalize(img_dataset_trans), vmax=1e-1, origin="lower")
        axes[i][0].set_title(f"Amplitude dataset {i+1}")
        axes[i][1].imshow(equalize(img_gen_trans), vmax=1e-1, origin="lower")
        axes[i][1].set_title(f"Amplitude generated {i+1}")

        if last:
            # Plot phase images
            axes[i][2].imshow(plot_phase(img_dataset_trans), cmap="hsv")
            axes[i][2].set_title(f"Phase dataset {i+1}")
            axes[i][3].imshow(plot_phase(img_gen_trans), cmap="hsv")
            axes[i][3].set_title(f"Phase generated {i+1}")

            # Compute Fourier transforms for amplitude and phase for each channel
            dataset_amplitude_ft, dataset_phase_vectors = (
                plot_fourier_transform_amplitude_phase(img_dataset)
            )
            generated_amplitude_ft, generated_phase_vectors = (
                plot_fourier_transform_amplitude_phase(img_gen)
            )

            for ch in range(num_channels):
                # Plot Fourier Transforms of the amplitude and phase for dataset and generated images
                base_index = 4 + ch * 4  # Base index for each channel's plots
                axes[i][base_index].imshow(dataset_amplitude_ft[ch], cmap="gray")
                axes[i][base_index].set_title(f"FT Amp Dataset Ch{ch+1}")
                axes[i][base_index + 1].imshow(generated_amplitude_ft[ch], cmap="gray")
                axes[i][base_index + 1].set_title(f"FT Amp Generated Ch{ch+1}")

                X, Y = np.meshgrid(
                    np.arange(samples[0][ch, :, :].shape[1]),
                    np.arange(samples[0][ch, :, :].shape[0]),
                )
                axes[i][base_index + 2].quiver(
                    X,
                    Y,
                    dataset_phase_vectors[ch][0],
                    dataset_phase_vectors[ch][1],
                    scale=60,
                )
                axes[i][base_index + 2].set_title(f"FT Phase Dataset Ch{ch+1}")
                axes[i][base_index + 3].quiver(
                    X,
                    Y,
                    generated_phase_vectors[ch][0],
                    generated_phase_vectors[ch][1],
                    scale=60,
                )
                axes[i][base_index + 3].set_title(f"FT Phase Generated Ch{ch+1}")

    # Remove axis labels for all plots
    for ax in axes.flatten():
        ax.axis("off")

    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close()


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
    train_indices = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 98] * 10
    valid_indices = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 98] * 10

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
