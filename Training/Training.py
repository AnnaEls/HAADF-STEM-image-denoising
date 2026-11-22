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

def compute_jacobian(
    model,
    noisy_image,
    row,
    col
):
    model.eval()

    x = noisy_image.detach().clone().requires_grad_(True)

    with torch.enable_grad():

        output = model(x)

        J = torch.autograd.grad(
            outputs=output[0, 0, row, col],
            inputs=x,
            retain_graph=True
        )[0]   

    return J



def compute_hybrid_jacobians(
    model,
    noisy_image,
    row,
    col
):
    model.eval()

    x = noisy_image.detach().clone().requires_grad_(True)

    with torch.enable_grad():

        # =====================================
        # SHARED ENCODER
        # =====================================
        features, skip_1 = model.encoder(x)
        x_shared = features
        x_afno = features
        x_cnn = features

        # =====================================
        # AFNO BRANCH: ONLY AFNO LAYERS
        # =====================================
        
        for blk in model.bottleneck_afno:
            x_afno = blk(x_afno, x_shared)

        # Restore H x W using shared skip
        y_afno = model.decoder_afno(
            x_afno,
            skip_1
        )

        output_afno = model.out_conv_afno(y_afno)


        # =====================================
        # CNN BRANCH: ONLY CNN LAYERS
        # =====================================
        cnn_skips = []

        # Additional CNN encoder levels
        for down in model.downs_cnn:

            x_cnn, skip = down(x_cnn)

            cnn_skips.append(skip)

        # CNN bottleneck
        x_cnn = model.bottleneck_cnn(x_cnn)

        # ------------------------------------------------------------
        # Decode CNN-specific levels
        # ------------------------------------------------------------

        for i, up in enumerate(model.ups_cnn):

            # Reverse skip order:
            # deepest skip first
            skip = cnn_skips[-(i + 1)]

            x_cnn = up(
                x_cnn,
                skip
            )

        # ------------------------------------------------------------
        # Final decoder uses SHARED first-encoder skip
        # ------------------------------------------------------------

        x_cnn = model.decoder_shared_cnn(
            x_cnn,
            skip_1
        )

        output_cnn = model.out_conv_cnn(x_cnn)


        # =====================================
        # AFNO INPUT-OUTPUT JACOBIAN
        # Encoder + AFNO + AFNO decoder
        # =====================================
        J_AFNO = torch.autograd.grad(
            outputs=output_afno[0, 0, row, col],
            inputs=x,
            retain_graph=True
        )[0]

        # =====================================
        # CNN INPUT-OUTPUT JACOBIAN
        # Encoder + CNN + CNN decoder
        # =====================================
        J_CNN = torch.autograd.grad(
            outputs=output_cnn[0, 0, row, col],
            inputs=x
        )[0]

    return (
        J_AFNO[0, 0].detach().cpu().numpy(),
        J_CNN[0, 0].detach().cpu().numpy(),
        output_afno[0, 0].detach().cpu().numpy(),
        output_cnn[0, 0].detach().cpu().numpy()
    )

def train_model(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42, jacobians = True, row = 64, col = 64):    
    
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

        if jacobians == True:
            J = compute_jacobian(model, input, row=row, col=col)
        else:
            model.eval()
        with torch.no_grad():
            denoised_image= model(input)
            tifffile.imwrite(f'{path}/{it+1:04d}.tif', convert(denoised_image.squeeze().detach().cpu().numpy()), imagej=True)
            if show_image:
               print(f"epoch {it + 1}, loss={loss.item():.6f}")
               plt.imshow(denoised_image.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               plt.show()
               if jacobians:
                   fig, axes = plt.subplots(1, 1, figsize=(6, 5))                
                   vmax = np.max(np.abs(J.detach().cpu().numpy()))                       
                   im1 = axes[0].imshow(J, cmap='seismic', vmin=-vmax, vmax=vmax)
                   axes[0].set_title("Jacobian")
                   axes[0].axis('off')
                   fig.colorbar(im1, ax=axes[0], label='Correlation')    
                   plt.tight_layout()
                   plt.show()  
        model.train()
    np.save(os.path.join(path, 'loss_history.npy'), np.array(loss_history))    

def train_hybrid_model(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42, jacobians = True, row = 64, col = 64):    
    
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

        if jacobians == True:
            J_AFNO, J_CNN, _ ,_ = compute_hybrid_jacobians(model, input, row=row, col=col)
        else:
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
               if jacobians:
                   fig, axes = plt.subplots(1, 2, figsize=(12, 5))                
                   vmax_afno = np.max(np.abs(J_AFNO))
                   vmax_cnn = np.max(np.abs(J_CNN)) if np.max(np.abs(J_CNN)) > 0 else 1e-6                
                   im1 = axes[0].imshow(J_AFNO, cmap='seismic', vmin=-vmax_afno, vmax=vmax_afno)
                   axes[0].set_title("J_AFNO (Fourier Branch)")
                   axes[0].axis('off')
                   fig.colorbar(im1, ax=axes[0], label='Correlation')                
                   im2 = axes[1].imshow(J_CNN, cmap='seismic', vmin=-vmax_cnn, vmax=vmax_cnn)
                   axes[1].set_title("J_CNN (CNN Branch)")
                   axes[1].axis('off')
                   fig.colorbar(im2, ax=axes[1], label='Correlation')                
                   plt.tight_layout()
                   plt.show() 
              
        model.train()
    
 
def train_hybrid_model_with_mask_offset(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, mask_offset = 3, show_image=False, seed=42, jacobians = True, row = 64, col = 64):    
    
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

        if jacobians == True:
            J_AFNO, J_CNN, _ ,_ = compute_hybrid_jacobians_explicit(modle, input, row=row, col=col)
        else:
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
            if jacobians:
               fig, axes = plt.subplots(1, 2, figsize=(12, 5))                
               vmax_afno = np.max(np.abs(J_AFNO))
               vmax_cnn = np.max(np.abs(J_CNN)) if np.max(np.abs(J_CNN)) > 0 else 1e-6                
               im1 = axes[0].imshow(J_AFNO, cmap='seismic', vmin=-vmax_afno, vmax=vmax_afno)
               axes[0].set_title("J_AFNO (Fourier Branch)")
               axes[0].axis('off')
               fig.colorbar(im1, ax=axes[0], label='Correlation')                
               im2 = axes[1].imshow(J_CNN, cmap='seismic', vmin=-vmax_cnn, vmax=vmax_cnn)
               axes[1].set_title("J_CNN (CNN Branch)")
               axes[1].axis('off')
               fig.colorbar(im2, ax=axes[1], label='Correlation')                
               plt.tight_layout()
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

