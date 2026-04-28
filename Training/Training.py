import torch
import torch.nn.functional as F

from Training.Masking import random_patch_mask
from Utilities.Utils import convert 

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

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

def train_model_self_guided(
    model, input, path, sigma, reg_coef,
    learning_rate=1e-3,
    z_learning_rate=1e-4,
    num_iter=1,
    patch_size=1,
    mask_ratio=0.2,
    tv_z_coef=1e-5,
    show_image=False,
    seed=42
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Fixed noisy target
    y = input.to(device)

    # Learnable input
    z = torch.nn.Parameter(y.clone())

    # Different learning rates for model and input
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": learning_rate},
        {"params": [z], "lr": z_learning_rate}
    ])

    def tv_loss(x):
        return (
            torch.mean(torch.abs(x[..., 1:, :] - x[..., :-1, :])) +
            torch.mean(torch.abs(x[..., :, 1:] - x[..., :, :-1]))
        )

    model.train()
    loss_history = []

    os.makedirs(path, exist_ok=True)

    for it in range(num_iter):

        # Two perturbed versions of learnable input
        z1 = z + sigma * torch.randn_like(z)
        z2 = z + sigma * torch.randn_like(z)

        masked_z1, mask = random_patch_mask(
            z1,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,
            epoch=it
        )

        masked_z2, _ = random_patch_mask(
            z2,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,
            epoch=it
        )

        output_1 = model(masked_z1)
        output_2 = model(masked_z2)

        # Reconstruction loss
        loss_recon = F.mse_loss(
            output_1 * (1 - mask),
            y * (1 - mask)
        )

        # Stability regularization
        loss_reg = torch.mean((output_1 - output_2.detach()) ** 2)

        # TV regularization on learnable input z
        loss_tv_z = tv_loss(z)

        loss = loss_recon + reg_coef * loss_reg + tv_z_coef * loss_tv_z

        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Prevent z from drifting too far
        with torch.no_grad():
            z.data = 0.99 * z.data + 0.01 * y.data
            z.clamp_(0.0, 1.0)

        model.eval()
        with torch.no_grad():
            denoised_image = model(z)

            tifffile.imwrite(
                f'{path}/{it+1:04d}.tif',
                convert(denoised_image.squeeze().cpu().numpy()),
                imagej=True
            )

            if show_image:
                print(
                    f"epoch {it + 1}, "
                    f"loss={loss.item():.6f}, "
                    f"recon={loss_recon.item():.6f}, "
                    f"reg={loss_reg.item():.6f}, "
                    f"tv_z={loss_tv_z.item():.6f}"
                )
                plt.imshow(denoised_image.squeeze().cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

        model.train()

    np.save(os.path.join(path, 'loss_history.npy'), np.array(loss_history))
