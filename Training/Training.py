import torch
import torch.nn.functional as F

from Training.Masking import random_patch_mask, random_patch_mask_with_offset
from Utilities.Utils import convert 

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import copy


def analyze_jacobian(
    model,
    noisy,
    row,
    col,
    local_radius=8,
    mask_target=False
):
    """
    noisy: [1, 1, H, W] tensor
    row, col: output pixel to analyze

    mask_target:
       False -> evaluate on complete noisy image
       True  -> zero the target pixel before evaluation
    """
    model.eval()

    x = noisy.detach().clone().requires_grad_(True)

    if mask_target:
        mask = torch.ones_like(x)
        mask[0, 0, row, col] = 0
        model_input = x * mask
    else:
        model_input = x

    with torch.enable_grad():
        output  = model(model_input)
        target = output[0][0, 0, row, col]

        grad = torch.autograd.grad(
            target, x, retain_graph=False
        )[0]

    J = grad[0, 0].detach().cpu().numpy()

    H, W = J.shape
    yy, xx = np.ogrid[:H, :W]

    distance = np.sqrt(
        (yy - row)**2 + (xx - col)**2
    )

    abs_J = np.abs(J)
    total = abs_J.sum() + 1e-12

    self_dependence = abs_J[row, col]

    local = abs_J[distance <= local_radius].sum()
    nonlocal_weight = abs_J[distance > local_radius].sum()

    metrics = {
        "self": float(self_dependence),
        "local_fraction": float(local / total),
        "nonlocal_fraction": float(nonlocal_weight / total),
        "total_sensitivity": float(total)
    }

    return J, metrics


import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_hybrid_jacobians(
    model,
    noisy_image,
    row,
    col,
    device="cuda"
):
    """
    Calculate input-output Jacobian maps for both branches.

    Parameters
    ----------
    model : torch.nn.Module
        Hybrid model returning (out_fourier, out_cnn).

    noisy_image : numpy.ndarray or torch.Tensor
        2D noisy image, or tensor [1, 1, H, W].

    row, col : int
        Coordinates of the selected output pixel.

    Returns
    -------
    J_fourier : ndarray [H, W]
    J_cnn : ndarray [H, W]
    out_fourier : ndarray [H, W]
    out_cnn : ndarray [H, W]
    """

    model = model.to(device)
    model.eval()

    if isinstance(noisy_image, np.ndarray):
        x = torch.as_tensor(
            noisy_image,
            dtype=torch.float32,
            device=device
        )
    else:
        x = noisy_image.detach().to(device)

    if x.ndim == 2:
        x = x[None, None]

    x = x.clone().requires_grad_(True)

    # Forward pass through shared encoder and both branches
    out_fourier, out_cnn = model(x)

    # Select same pixel in both reconstructed images
    pixel_fourier = out_fourier[0, 0, row, col]
    pixel_cnn = out_cnn[0, 0, row, col]

    # Fourier branch Jacobian row
    grad_fourier = torch.autograd.grad(
        outputs=pixel_fourier,
        inputs=x,
        retain_graph=True
    )[0]

    # CNN branch Jacobian row
    grad_cnn = torch.autograd.grad(
        outputs=pixel_cnn,
        inputs=x,
        retain_graph=False
    )[0]

    J_fourier = grad_fourier[0, 0].detach().cpu().numpy()
    J_cnn = grad_cnn[0, 0].detach().cpu().numpy()

    out_fourier = out_fourier[0, 0].detach().cpu().numpy()
    out_cnn = out_cnn[0, 0].detach().cpu().numpy()

    return J_fourier, J_cnn, out_fourier, out_cnn


def train_model(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42):    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input = input.to(device)
    model.train()

    loss_history = []

    os.makedirs(path, exist_ok=True)

    for it in range(num_iter):   
        masked_input, mask = random_patch_mask(
            input,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,         
            epoch=it            
        )

        output_afno = model(masked_input)

        loss = F.mse_loss(output_afno * (1 - mask), input * (1 - mask))
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            denoised_image= model(input)
            tifffile.imwrite(f'{path}/{it+1:04d}.tif', convert(denoised_image.squeeze().detach().cpu().numpy()), imagej=True)
            if show_image:
               print(f"epoch {it + 1}, loss={loss.item():.6f}")
               plt.imshow(denoised_image.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               plt.show()
        model.train()
    np.save(os.path.join(path, 'loss_history.npy'), np.array(loss_history))    

def train_hybrid_model(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, row = 55, col = 79, seed=42):    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input = input.to(device)
    model.train()

    loss_history = []

    os.makedirs(path, exist_ok=True)

    for it in range(num_iter):   
        masked_input, mask = random_patch_mask(
            input,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,         
            epoch=it            
        )

        output_afno, output_cnn = model(masked_input)

        loss = F.mse_loss(output_afno * (1 - mask), input * (1 - mask)) + F.mse_loss(output_cnn * (1 - mask), input * (1 - mask))
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            denoised_image_afno, denoised_image_cnn = model(input)
            tifffile.imwrite(f'{path}/AFNO_{(it+1):04d}.tif', convert(denoised_image_afno.squeeze().detach().cpu().numpy()), imagej=True)
            tifffile.imwrite(f'{path}/CNN_{(it+1):04d}.tif', convert(denoised_image_cnn.squeeze().detach().cpu().numpy()), imagej=True)
            if show_image:
               print(f"epoch {it + 1}, loss={loss.item():.6f}")
               plt.figure(figsize=(8, 8)) # Create a new figure for the 4 plots
               plt.subplot(2,2,1)
               plt.imshow(denoised_image_afno.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               plt.subplot(2,2,2)
               plt.imshow(denoised_image_cnn.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               #plt.subplot(2,2,3)
               #plt.imshow(denoised_image_fused.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();

               plt.show()
               J_F, J_C, img_F, img_C = compute_hybrid_jacobians(model, input, row=row, col=col, device=device)
               fig, axes = plt.subplots(2, 2, figsize=(10, 9))

               # Reconstructions
               axes[0, 0].imshow(img_F, cmap="gray")
               axes[0, 0].plot(col, row, "r+", markersize=10)
               axes[0, 0].set_title("Fourier branch reconstruction")
                
               axes[0, 1].imshow(img_C, cmap="gray")
               axes[0, 1].plot(col, row, "r+", markersize=10)
               axes[0, 1].set_title("CNN branch reconstruction")
                
               # Shared color scale
               vmax = max(np.max(np.abs(J_F)), np.max(np.abs(J_C)))
                
               im1 = axes[1, 0].imshow(
                    J_F, cmap="seismic",
                    vmin=-vmax, vmax=vmax
                )
               axes[1, 0].set_title("Fourier branch Jacobian") 
        model.train()
    
 
def train_hybrid_model_with_mask_offset(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, mask_offset = 3, show_image=False, seed=42):    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input = input.to(device)
    model.train()

    loss_history = []

    os.makedirs(path, exist_ok=True)

    for it in range(num_iter):   
        masked_input, mask = random_patch_mask_with_offset(
            input,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            offset = mask_offset,
            seed=seed,         
            epoch=it            
        )

        output_afno, output_cnn = model(masked_input)

        loss = F.mse_loss(output_afno * (1 - mask), input * (1 - mask)) + F.mse_loss(output_cnn * (1 - mask), input * (1 - mask))
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            denoised_image_afno, denoised_image_cnn = model(input)
            tifffile.imwrite(f'{path}/AFNO_{(it+1):04d}.tif', convert(denoised_image_afno.squeeze().detach().cpu().numpy()), imagej=True)
            tifffile.imwrite(f'{path}/CNN_{(it+1):04d}.tif', convert(denoised_image_cnn.squeeze().detach().cpu().numpy()), imagej=True)
            if show_image:
               print(f"epoch {it + 1}, loss={loss.item():.6f}")
               plt.figure(figsize=(8, 8)) # Create a new figure for the 4 plots
               plt.subplot(2,2,1)
               plt.imshow(denoised_image_afno.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               plt.subplot(2,2,2)
               plt.imshow(denoised_image_cnn.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               #plt.subplot(2,2,3)
               #plt.imshow(denoised_image_fused.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();

               plt.show()
        model.train()
    np.save(os.path.join(path, 'loss_history.npy'), np.array(loss_history))        


    

def masked_mse(pred, target, target_mask):
    """
    MSE calculated ONLY on selected target pixels.

    target_mask:
        True = pixel included in loss
    """

    target_mask = target_mask.expand_as(pred)

    return (
        (pred - target) ** 2
    )[target_mask].mean()

