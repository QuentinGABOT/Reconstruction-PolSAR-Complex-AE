import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import logging
import random
from scipy.linalg import eigh
from numpy import linalg as LA
import os
import glob
import shutil

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

        transformed_amplitude = (torch.log10(amplitude) - np.log10(m)) / (
            np.log10(M) - np.log10(m)
        )
        # Recombine to form new complex tensor
        new_tensor = transformed_amplitude * torch.exp(1j * phase)

        return new_tensor


"""
class ExpAmplitudeTransform:
    def __init__(self, characteristics):
        # Store the channel characteristics
        self.characteristics = characteristics

    def __call__(self, tensor):
        new_tensor = tensor
        m = 2e-2
        M = 40

        amplitude = torch.abs(tensor)
        phase = torch.angle(tensor)

        inv_transformed_amplitude = torch.exp(
            (np.log10(M) - np.log10(m)) * amplitude + np.log10(m)
        )
        # Recombine to form new complex tensor
        new_tensor = inv_transformed_amplitude * torch.exp(1j * phase)

        return new_tensor
"""
# transfo Pauli, Cameron, Krogager


def exp_amplitude_transform(tensor):
    tensor = torch.from_numpy(tensor)
    m = 2e-2
    M = 40

    amplitude = torch.abs(tensor)
    phase = torch.angle(tensor)

    inv_transformed_amplitude = torch.clip(
        torch.exp(((np.log10(M) - np.log10(m)) * amplitude + np.log10(m)) * np.log(10)),
        0,
        10**9,
    )

    # Recombine to form new complex tensor
    new_tensor = inv_transformed_amplitude * torch.exp(1j * phase)

    return new_tensor


def equalize(image, p2=None, p98=None):
    """
    Automatically adjust contrast of the SAR image
    Input: intensity or amplitude in dB scale
    """
    img = np.log10(np.abs(image))
    if not p2:
        p2, p98 = np.percentile(img, (2, 98))
    img_resc = np.round(
        exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1)) * 255
    ).astype(np.uint8)

    return img_resc, (p2, p98)


def angular_distance(image1, image2):
    """
    Compute the angular distance between two phase angles, phase1 and phase2, with results in [-pi, pi].
    """
    diff = np.angle(image1) - np.angle(image2) + np.pi
    angular_dist = np.mod(diff, 2 * np.pi) - np.pi
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


########################################################################################################################
### CALCULATE THE MEANS OF THE CLASSES AFTER H-alpha INITIALIZATION
########################################################################################################################

### Calculate the means of the classes. This function will also be used below later, after a continuous update of classes.
### As input it takes the image of stacked up covariances, and a mask of classes.


def calculate_means_of_classes(image_of_stacked_covariances, classes_H_alpha):

    list_of_classes = [1, 2, 4, 5, 6, 7, 8, 9]  ### class 3 is not possible
    dictionary_of_means = {}

    for k in list_of_classes:
        ### Create a mask for the image which is TRUE where a pixel belongs to the class, and FALSE otherwise.
        mask = classes_H_alpha == k
        size_of_mask_1, size_of_mask_2 = mask.shape
        mask = np.reshape(mask, (size_of_mask_1, size_of_mask_2, 1))
        ### Multiply image with mask, i.e. all pixels/covariances not belonging to the initial class k will be set to zero.
        cov_times_mask = image_of_stacked_covariances * mask
        cov_times_mask = np.reshape(
            cov_times_mask,
            (
                cov_times_mask.shape[0] * cov_times_mask.shape[1],
                cov_times_mask.shape[2],
            ),
        )
        ### Since all other entries are set to zero, taking the mean over the whole image equals the mean of just class k.
        mean_of_class = np.mean(cov_times_mask, axis=0)
        dictionary_of_means["mean" + str(k)] = mean_of_class

    return dictionary_of_means


def h_alpha(pauli_radar_image):
    ########################################################################################################################
    fullsamples = pauli_radar_image
    s1, s2, p = fullsamples.shape
    son = 7
    ########################################################################################################################

    ### Create variables that will be used for the H-alpha decomposition.
    p_vector = np.zeros(3)
    alpha_vector = np.zeros(3)
    H_alpha = np.zeros((s1 - (son - 1), s2 - (son - 1), 2))

    ### This will contain the original classes after the H alpha initialization.
    classes_H_alpha_original = np.zeros((s1 - (son - 1), s2 - (son - 1)))

    ### This is the image containing as pixels the local covariances 'stacked up'.
    ### Since a part of the edge is lost, it is slightly smaller than the original image.
    covariances_stacked = np.zeros(
        (s1 - (son - 1), s2 - (son - 1), p * p), dtype=complex
    )

    for k in range(s1 - (son - 1)):
        for l in range(s2 - (son - 1)):
            ###### calculate the local empirical covariance matrix (or second moment) of the pixel and neighborhood under consideration
            local_data_matrix = np.reshape(
                fullsamples[k : k + son, l : l + son, :], (son**2, p)
            )
            local_covariance = np.dot(
                np.conjugate(local_data_matrix).T, local_data_matrix
            ) / (son**2)
            ##### stack up the covariance matrices in one large vector
            local_covariance_stacked = np.reshape(local_covariance, (1, 1, p * p))
            covariances_stacked[k, l, :] = local_covariance_stacked
            ##### spectral decomposition of the local covariance - calculate H and alpha values for each pixel
            eigenvalues, eigenvectors = eigh(local_covariance)
            D = np.diag(eigenvalues)
            U = eigenvectors
            # spectral_dec = reduce(np.dot, [U, D, np.conj(U).T])
            ##### Calculate the H alpha decomposition - i.e. the initialization of the classes!
            for i in range(3):
                p_vector[i] = eigenvalues[i] / np.sum(eigenvalues)
                alpha_vector[i] = np.arccos(abs(eigenvectors[0, i]))
            H = -np.dot(p_vector, np.log(p_vector))
            alpha = np.dot(p_vector, alpha_vector) * (180.0 / 3.14159)
            H_alpha[k, l, 0] = H
            H_alpha[k, l, 1] = alpha

            ### The original class assigned to the pixels via the initial H - alpha decomposition is simply determined by a distinction of the different cases.
            if H <= 0.5:
                if alpha <= 42.5:
                    classes_H_alpha_original[k, l] = 9
                elif alpha <= 47.5:
                    classes_H_alpha_original[k, l] = 8
                elif alpha <= 90:
                    classes_H_alpha_original[k, l] = 7
            elif H <= 0.9:
                if alpha <= 40:
                    classes_H_alpha_original[k, l] = 6
                elif alpha <= 50:
                    classes_H_alpha_original[k, l] = 5
                elif alpha <= 90:
                    classes_H_alpha_original[k, l] = 4
            elif H <= 1.0:
                if alpha <= 55:
                    classes_H_alpha_original[k, l] = 2
                elif alpha <= 90:
                    classes_H_alpha_original[k, l] = 1

    ### Create the data matrix by reshaping, which contains all covariances as rows.
    X1 = np.reshape(covariances_stacked, ((s1 - (son - 1)) * (s2 - (son - 1)), p * p))

    return classes_H_alpha_original


def show_images(samples, generated, image_path, last):
    num_samples = len(samples)
    num_channels = samples[0].shape[
        0
    ]  # Assuming 3 channels for PolSAR images: HH, HV(=VH), VV

    if last:
        ncols = (
            9 + 4 * num_channels
        )  # Adjusted to include an extra column for the MSE histogram
    else:
        ncols = 9  # Amplitude images, and one column for the MSE histogram

    fig, axes = plt.subplots(
        nrows=num_samples, ncols=ncols, figsize=(5 * ncols, 5 * num_samples)
    )
    axes = np.atleast_2d(axes)  # Ensure axes is a 2D array for consistency
    channels = ["HH-VV", "2HV", "HH+VV"]

    for i in range(num_samples):
        idx = 0
        img_dataset, img_gen = (
            exp_amplitude_transform(samples[i]).numpy(),
            exp_amplitude_transform(generated[i]).numpy(),
        )
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

        # Compute pixel-wise amplitude difference and plot histogram in the same figure
        mse_values = np.abs(img_dataset_trans) - np.abs(
            img_gen_trans
        )  # we don't use the equalize output due to the transform applied to the amplitude

        axes[i][idx].hist(mse_values.flatten(), bins=100, alpha=0.75)
        axes[i][idx].set_title(f"Amplitude Difference Histogram {i+1}")
        axes[i][idx].set_xlabel("Amplitude Difference Value")
        axes[i][idx].set_ylabel("Frequency")
        idx += 1

        for ch in range(num_channels):
            axes[i][idx].imshow(
                plot_angular_distance(
                    img_dataset_trans[:, :, ch], img_gen_trans[:, :, ch]
                ),
                cmap="hsv",
                origin="lower",
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

        ### Plot the H - alpha initialization, i.e. the mask of classes assigend to the pixels according to the H - alpha decomposition.
        axes[i][idx].imshow(
            h_alpha(img_dataset_trans), origin="lower", cmap="tab10", vmin=1, vmax=9
        )
        axes[i][idx].set_title(f"H_alpha dataset {i+1}")
        axes[i][idx].axis("off")  # Turn off axes for image plot
        idx += 1

        axes[i][idx].imshow(
            h_alpha(img_gen_trans), origin="lower", cmap="tab10", vmin=1, vmax=9
        )
        axes[i][idx].set_title(f"H_alpha generated {i+1}")
        axes[i][idx].axis("off")  # Turn off axes for image plot
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


def delete_folders_with_few_pngs(min_png_count=10, root_path=None):
    """
    Deletes folders under `root_path` containing fewer than `min_png_count` .png files.

    :param root_path: Path to the directory to search through.
    :param min_png_count: Minimum number of .png files a folder must contain to be kept.
    """
    if root_path is None:
        root_path = (
            "/home/qgabot/Documents/complex-valued-generarive-ai-for-sar-imaging/logs"
        )
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            png_files = [
                file for file in os.listdir(folder_path) if file.endswith(".png")
            ]
            if len(png_files) < min_png_count:
                print(
                    f"Deleting folder: {folder_path} (contains {len(png_files)} .png files)"
                )
                shutil.rmtree(folder_path)
