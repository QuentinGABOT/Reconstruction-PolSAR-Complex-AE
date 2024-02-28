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


class LogAmplitudeTransform:
    def __init__(self, characteristics):
        # Store the channel characteristics
        self.characteristics = characteristics

    def __call__(self, tensor):
        new_tensor = tensor
        m = 2e-2
        M = 40

        amplitude = torch.clip(torch.abs(tensor), m, M)
        phase = torch.angle(tensor)

        for c in range(tensor.shape[0]):

            transformed_amplitude = (torch.log10(amplitude[c]) - np.log10(m)) / (
                np.log10(M) - np.log10(m)
            )
            # Recombine to form new complex tensor
            new_tensor[c, :, :] = transformed_amplitude * torch.exp(1j * phase[c])

        return new_tensor


def equalize(image, p2=None, p98=None):
    """
    Automatically adjust contrast of the SAR image
    Input: intensity or amplitude in dB scale
    """
    img = np.abs(image)
    if not p2:
        p2, p98 = np.percentile(img, (2, 98))
    img_resc = np.round(
        exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1)) * 255
    ).astype(np.uint8)

    return img_resc, (p2, p98)


def angular_distance(phase1, phase2):
    """
    Compute the angular distance between two phase angles, phase1 and phase2, with results in [-pi, pi].
    """
    diff = np.angle(phase1) - np.angle(phase2)
    angular_dist = np.arctan2(np.sin(diff), np.cos(diff))
    return angular_dist


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


def plot_angular_distance(image1, image2):
    """
    Plot the phase of a PolSAR image and normalize it to [0, 255].
    """
    ang_distance_image = angular_distance(image1, image2)
    # Normalize phase to [0, 1]
    normalized_ang_distance_image = (ang_distance_image + np.pi) / (2 * np.pi)
    # Scale to [0, 255] and convert to integer
    scaled_ang_distance_image = np.round(normalized_ang_distance_image * 255).astype(
        np.uint8
    )
    return scaled_ang_distance_image


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


def show_images(samples, generated, image_path, last):
    num_samples = len(samples)
    num_channels = samples[0].shape[
        0
    ]  # Assuming 3 channels for PolSAR images: HH, HV(=VH), VV

    if last:
        ncols = (
            7 + 4 * num_channels
        )  # Adjusted to include an extra column for the MSE histogram
    else:
        ncols = 7  # Amplitude images, and one column for the MSE histogram

    fig, axes = plt.subplots(
        nrows=num_samples, ncols=ncols, figsize=(5 * ncols, 5 * num_samples)
    )
    axes = np.atleast_2d(axes)  # Ensure axes is a 2D array for consistency
    channels = ["HH-VV", "2HV", "HH+VV"]

    for i in range(num_samples):
        idx = 0
        img_dataset, img_gen = samples[i], generated[i]
        img_dataset_trans = img_dataset.transpose(1, 2, 0)
        img_gen_trans = img_gen.transpose(1, 2, 0)

        # Plot amplitude images
        eq_dataset, (p2, p98) = equalize(img_dataset_trans)
        axes[i][idx].imshow(eq_dataset, origin="lower")
        axes[i][idx].set_title(f"Amplitude dataset {i+1}")
        axes[i][idx].axis("off")  # Turn off axes for image plot
        idx += 1

        eq_generated, _ = equalize(img_gen_trans, p2=p2, p98=p98)
        axes[i][idx].imshow(eq_generated, origin="lower")
        axes[i][idx].set_title(f"Amplitude generated {i+1}")
        axes[i][idx].axis("off")  # Turn off axes for image plot
        idx += 1

        # Compute pixel-wise MSE and plot histogram in the same figure
        mse_values = np.abs(img_dataset_trans) - np.abs(
            img_gen_trans
        )  # we don't use the equalize output due to the transform applied to the amplitude
        axes[i][idx].hist(mse_values.flatten(), bins=100, alpha=0.75)
        axes[i][idx].set_title(f"MSE Histogram {i+1}")
        axes[i][idx].set_xlabel("MSE Value")
        axes[i][idx].set_ylabel("Frequency")
        idx += 1

        """
        # Plot phase images
        axes[i][idx].imshow(plot_phase(img_dataset_trans), cmap="hsv")
        axes[i][idx].set_title(f"Phase dataset {i+1}")
        axes[i][idx].axis("off")  # Turn off axes for image plot
        idx += 1

        axes[i][idx].imshow(plot_phase(img_gen_trans), cmap="hsv")
        axes[i][idx].set_title(f"Phase generated {i+1}")
        axes[i][idx].axis("off")  # Turn off axes for image plot
        idx += 1
        """

        for ch in range(num_channels):
            axes[i][idx].imshow(
                plot_angular_distance(
                    img_dataset_trans[:, :, ch], img_gen_trans[:, :, ch]
                ),
                cmap="hsv",
            )
            axes[i][idx].set_title(f"Angular Distance pixel-wise {i+1} " + channels[ch])
            axes[i][idx].axis("off")  # Turn off axes for image plot
            idx += 1

        # Plot histogram of angular distances for phase images
        axes[i][idx].hist(
            angular_distance(img_dataset_trans, img_gen_trans).flatten(),
            bins=100,
            alpha=0.75,
        )
        axes[i][idx].set_title(f"Angular Distance Histogram {i+1}")
        axes[i][idx].set_xlabel("Angular Distance (radians)")
        axes[i][idx].set_ylabel("Frequency")
        idx += 1

        # If last, continue with the original functionality for phase and FT amplitude images
        if last:
            # Compute Fourier transforms for amplitude and phase for each channel
            dataset_amplitude_ft, dataset_phase_vectors = (
                plot_fourier_transform_amplitude_phase(img_dataset)
            )
            generated_amplitude_ft, generated_phase_vectors = (
                plot_fourier_transform_amplitude_phase(img_gen)
            )

            for ch in range(num_channels):
                # Plot Fourier Transforms of the amplitude and phase for dataset and generated images
                # base_index = 5 + ch * 4  # Base index for each channel's plots
                axes[i][idx].imshow(dataset_amplitude_ft[ch], cmap="gray")
                axes[i][idx].set_title(f"FT Amp Dataset " + channels[ch])
                axes[i][idx].axis("off")  # Turn off axes for image plot
                idx += 1

                axes[i][idx].imshow(generated_amplitude_ft[ch], cmap="gray")
                axes[i][idx].set_title(f"FT Amp Generated " + channels[ch])
                axes[i][idx].axis("off")  # Turn off axes for image plot
                idx += 1

                X, Y = np.meshgrid(
                    np.arange(samples[0][ch, :, :].shape[1]),
                    np.arange(samples[0][ch, :, :].shape[0]),
                )

                axes[i][idx].quiver(
                    X,
                    Y,
                    dataset_phase_vectors[ch][0],
                    dataset_phase_vectors[ch][1],
                    scale=60,
                )
                axes[i][idx].set_title(f"FT Phase Dataset " + channels[ch])
                axes[i][idx].axis("off")  # Turn off axes for image plot
                idx += 1

                axes[i][idx].quiver(
                    X,
                    Y,
                    generated_phase_vectors[ch][0],
                    generated_phase_vectors[ch][1],
                    scale=60,
                )
                axes[i][idx].set_title(f"FT Phase Generated " + channels[ch])
                axes[i][idx].axis("off")  # Turn off axes for image plot
                idx += 1

    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    # Assuming PolSFAlos2Dataset and data_config are defined elsewhere
    input_transform = LogAmplitudeTransform(data_config["characteristics"])
    base_dataset = PolSFAlos2Dataset(
        root=data_config["trainpath"], download=False, transform=input_transform
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    # indices = list(range(len(base_dataset)))
    valid_indices = [79, 82, 84, 86, 88]
    indices = list(range(100))
    train_indices = [num for num in indices if num not in valid_indices]
    valid_indices = valid_indices
    # random.shuffle(indices)
    # num_valid = int(valid_ratio * len(base_dataset))
    # num_valid = int(valid_ratio * len(indices))
    # train_indices = indices[num_valid:]
    # valid_indices = indices[:num_valid]
    # train_indices = [78, 98] * 500
    # valid_indices = [78, 98] * 500
    # train_indices = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 98] * 10
    # valid_indices = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 98] * 10

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
