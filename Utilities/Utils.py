import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from skimage import exposure

def z_score_normalize(img):
    """
    Z-score normalizes a numpy image (any shape).
    Output will have mean 0 and std 1.
    """
    mean = np.mean(img)
    std = np.std(img)
    # Prevent division by zero
    if std == 0:
        std = 1
    return (img - mean) / std

def prepare_input(path, show_image = True):
  noisy_image = np.array(tifffile.imread(path))
  noisy_image_tensor = torch.from_numpy(z_score_normalize(noisy_image)).unsqueeze(0).unsqueeze(0).float()
  if show_image:
     plt.imshow(noisy_image_tensor[0,0], cmap='gray'); plt.axis('off'); plt.tight_layout(); plt.show();
  return noisy_image_tensor

def prepare_input_amp_phase(path, show_image=True, normalize_separately=True, clip_limit = 0.2):
    """
    Loads image and creates a 3-channel tensor:

        channel 0: original image
        channel 1: inverse FFT reconstructed from amplitude only
        channel 2: inverse FFT reconstructed from phase only

    Output:
        noisy_image_tensor shape = [1, 3, H, W]
    """

    # Load image
    noisy_image = np.array(tifffile.imread(path)).astype(np.float32)

    # Convert RGB/RGBA to grayscale if needed
    if noisy_image.ndim == 3:
        noisy_image = noisy_image[..., :3].mean(axis=2)

    # Fourier transform
    F = np.fft.fft2(noisy_image)

    amplitude = np.abs(F)
    phase = np.angle(F)

    # Inverse FFT from amplitude only
    # Equivalent to setting phase = 0
    F_amp_only = amplitude * np.exp(1j * 0.0)
    image_amp_only = np.fft.ifft2(F_amp_only).real.astype(np.float32)

    # Inverse FFT from phase only
    # Equivalent to setting amplitude = 1
    F_phase_only = np.exp(1j * phase)
    image_phase_only = np.fft.ifft2(F_phase_only).real.astype(np.float32)

    # Normalize
    if normalize_separately:
        noisy_image_norm = z_score_normalize(noisy_image)
        image_amp_only_norm = z_score_normalize(image_amp_only)
        image_phase_only_norm = z_score_normalize(image_phase_only)

        stacked = np.stack(
            [
                noisy_image_norm,
                image_amp_only_norm,
                image_phase_only_norm,
            ],
            axis=0
        )

    else:
        stacked_raw = np.stack(
            [
                noisy_image,
                image_amp_only,
                image_phase_only,
            ],
            axis=0
        )

        stacked = z_score_normalize(stacked_raw)

    # [C, H, W] -> [1, C, H, W]
    noisy_image_tensor = torch.from_numpy(stacked).unsqueeze(0).float()

    if show_image:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(noisy_image_tensor[0, 0].cpu(), cmap="gray")
        axes[0].set_title("Original\nz-score")

        img_amp_disp = exposure.rescale_intensity(image_amp_only_norm, out_range=(0, 1))
        img_amp_disp = exposure.equalize_adapthist(img_amp_disp, clip_limit=clip_limit)

        axes[1].imshow(img_amp_disp, cmap="gray")
        axes[1].set_title("Amplitude-only IFFT\nz-score")

        axes[2].imshow(noisy_image_tensor[0, 2].cpu(), cmap="gray")
        axes[2].set_title("Phase-only IFFT\nz-score")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    return noisy_image_tensor

def convert(image):
  image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
  return image
