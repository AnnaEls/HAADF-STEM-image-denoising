import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from scipy.fft import fft2
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from Utilities.Utils import convert

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
    stopping_iter =  iterations[np.argmin(fourier_entropies)]
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

def calculate_metrics(path, path_to_clean_image, show_graphs=False):
    clean_image = np.array(tifffile.imread(path_to_clean_image))
    # Apply the same conversion to clean_image for consistent comparison
    clean_image_converted = convert(clean_image) # This will be uint8 0-255

    psnrs_afno = []
    ssims_afno = []
    psnrs_cnn = []
    ssims_cnn = []
    iterations = []
    image_files_afno = sorted([f for f in os.listdir(path) if f.endswith('.tif') and f.startswith('AFNO_')])
    image_files_cnn = sorted([f for f in os.listdir(path) if f.endswith('.tif') and f.startswith('CNN_')])

    # Ensure both lists have the same number of files for zipping
    min_len = min(len(image_files_afno), len(image_files_cnn))
    image_files_afno = image_files_afno[:min_len]
    image_files_cnn = image_files_cnn[:min_len]

    for image_file_afno, image_file_cnn in zip(image_files_afno, image_files_cnn):
        denoised_image_afno = np.array(tifffile.imread(os.path.join(path, image_file_afno))) # This is uint8 0-255
        denoised_image_cnn = np.array(tifffile.imread(os.path.join(path, image_file_cnn)))   # This is uint8 0-255
        iteration = int(image_file_afno[5:9]) # Assuming format AFNO_XXXX.tif, number starts at index 5

        # Now both clean_image_converted and denoised_images are uint8 0-255
        current_psnr_afno = psnr_metric(clean_image_converted, denoised_image_afno, data_range=255)
        current_ssim_afno = ssim_metric(clean_image_converted, denoised_image_afno, data_range=255)
        current_psnr_cnn = psnr_metric(clean_image_converted, denoised_image_cnn, data_range=255)
        current_ssim_cnn = ssim_metric(clean_image_converted, denoised_image_cnn, data_range=255)

        psnrs_afno.append(current_psnr_afno)
        ssims_afno.append(current_ssim_afno)
        psnrs_cnn.append(current_psnr_cnn)
        ssims_cnn.append(current_ssim_cnn)
        iterations.append(iteration)

    if show_graphs:
        sorted_indices = np.argsort(iterations)
        iterations = np.array(iterations)[sorted_indices]
        psnrs_afno = np.array(psnrs_afno)[sorted_indices]
        ssims_afno = np.array(ssims_afno)[sorted_indices]
        psnrs_cnn = np.array(psnrs_cnn)[sorted_indices]
        ssims_cnn = np.array(ssims_cnn)[sorted_indices]

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.plot(iterations, psnrs_afno)
        plt.xlabel('Iterations')
        plt.ylabel('PSNR AFNO branch')
        plt.title('PSNR vs. Iterations (AFNO)')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(iterations, ssims_afno)
        plt.xlabel('Iterations')
        plt.ylabel('SSIM AFNO branch')
        plt.title('SSIM vs. Iterations (AFNO)')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(iterations, psnrs_cnn)
        plt.xlabel('Iterations')
        plt.ylabel('PSNR CNN branch')
        plt.title('PSNR vs. Iterations (CNN)')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(iterations, ssims_cnn)
        plt.xlabel('Iterations')
        plt.ylabel('SSIM CNN branch')
        plt.title('SSIM vs. Iterations (CNN)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    max_psnr_afno = np.max(psnrs_afno)
    max_ssim_afno = np.max(ssims_afno)
    max_psnr_cnn = np.max(psnrs_cnn)
    max_ssim_cnn = np.max(ssims_cnn)
    best_psnr_iteration_afno = iterations[np.argmax(psnrs_afno)]
    best_ssim_iteration_afno = iterations[np.argmax(ssims_afno)]
    np.save(os.path.join(path, 'psnrs_afno.npy'), psnrs_afno)
    np.save(os.path.join(path, 'ssims_afno.npy'), ssims_afno)
    np.save(os.path.join(path, 'psnrs_cnn.npy'), psnrs_cnn)
    np.save(os.path.join(path, 'ssims_cnn.npy'), ssims_cnn)
    return max_psnr_afno, max_ssim_afno, best_psnr_iteration_afno, best_ssim_iteration_afno, max_psnr_cnn, max_ssim_cnn
