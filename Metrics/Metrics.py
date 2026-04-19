import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from scipy.fft import fft2
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from Utilities.Utilities import convert

def fourier_entropy(image, normalize=True, remove_dc=True, eps=1e-12):
    """
    Fourier (spectral) entropy of a 2D image.
    """

    if image.ndim != 2:
        raise ValueError("Input must be a 2D array")

    if remove_dc:
        image = image - np.mean(image)

    # Power spectrum
    power = np.abs(fft2(image))**2

    # Remove DC explicitly 
    if remove_dc:
        power[0, 0] = 0.0

    # Probability distribution
    p = power.ravel()
    p = p + eps
    p /= p.sum()

    # Shannon entropy
    H = entropy(p, base=2)

    # Normalize entropy
    if normalize:
        H /= np.log2(p.size)

    return H

def calculate_metrics(path, path_to_clean_image, show_graphs=False):
    clean_image_raw = np.array(tifffile.imread(path_to_clean_image))
    clean_image = convert(clean_image_raw)

    psnrs = []
    ssims = []
    iterations = []
    fourier_entropies = [] 
  
    image_files = sorted([f for f in os.listdir(path) if f.endswith('.tif') and f[:-4].isdigit()])

    if not image_files: 
        print("No .tif files found in the specified directory.")
        return None, None, None, None # Return None for all metrics if no files are found

    for image_file in image_files:
        iteration = int(image_file[:-4])
        denoised_image = np.array(tifffile.imread(os.path.join(path, image_file)))
        
        current_psnr = psnr_metric(clean_image, denoised_image, data_range=255)
        current_ssim = ssim_metric(clean_image, denoised_image, data_range=255)
        psnrs.append(current_psnr)
        ssims.append(current_ssim)      
        f_entropy = fourier_entropy(denoised_image)
        fourier_entropies.append(f_entropy)
        iterations.append(iteration)

    max_psnr = np.max(psnrs)
    max_ssim = np.max(ssims)
    best_psnr_iteration = iterations[np.argmax(psnrs)]
    best_ssim_iteration = iterations[np.argmax(ssims)]
    stopping_iter = np.min(fourier_entropies)
    np.save(os.path.join(path, 'psnrs.npy'), psnrs)
    np.save(os.path.join(path, 'ssims.npy'), ssims)
    np.save(os.path.join(path, 'fourier_entropy.npy'), fourier_entropies )

    if show_graphs:
        sorted_indices = np.argsort(iterations)
        iterations = np.array(iterations)[sorted_indices]
        psnrs = np.array(psnrs)[sorted_indices]
        ssims = np.array(ssims)[sorted_indices]
        fourier_entropies = np.array(fourier_entropies)[sorted_indices]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(iterations, psnrs)
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.title('PSNR vs. Epochs')
        plt.grid(True)        
        plt.axvline(x=stopping_iter,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f"Stop @ {stopping_iter}"
                    )


        plt.subplot(1, 3, 2)
        plt.plot(iterations, ssims)
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.title('SSIM vs. Epochs')
        plt.grid(True)        
        plt.axvline(x=stopping_iter,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f"Stop @ {stopping_iter}"
                    )

        plt.subplot(1, 3, 3)
        plt.plot(iterations, fourier_entropies)
        plt.xlabel('Epochs')
        plt.ylabel('Fourier Entropy')
        plt.title('Fourier Entropy vs. Epochs')
        plt.grid(True)
        if stopping_iter is not None:
          plt.axvline(x=stopping_iter,
                      color='red',
                      linestyle='--',
                      linewidth=2,
                      label=f"Stop @ {stopping_iter}"
                     )

        plt.tight_layout()

        plt.show()
    
    return max_psnr, max_ssim, best_psnr_iteration, best_ssim_iteration, stopping_iter
