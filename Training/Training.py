import torch
import torch.nn.functional as F

from Training.Masking import random_patch_mask
from Utilities.Utils import convert 

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


@torch.no_grad()
def sure_mc(model, y, sigma, eps=1e-3, num_mc=1):
    """
    Monte Carlo SURE estimate for Gaussian denoising.

    Parameters
    ----------
    model : torch.nn.Module
        Denoising model. It must take y as input and return denoised output.
    y : torch.Tensor
        Noisy image, shape [B, C, H, W].
    sigma : float
        Noise standard deviation in the same intensity scale as y.
    eps : float
        Small finite-difference perturbation.
    num_mc : int
        Number of Monte Carlo samples for divergence estimation.

    Returns
    -------
    sure_value : float
        SURE estimate per pixel.
    mse_term : float
        Residual term per pixel.
    div_term : float
        Divergence term per pixel.
    """

    model.eval()

    y = y.detach()
    f_y = model(y)

    num_pixels = y.numel()

    # Residual term: ||f(y) - y||^2 / N
    mse_term = torch.sum((f_y - y) ** 2) / num_pixels

    div_estimates = []

    for _ in range(num_mc):
        # Rademacher random vector: values are -1 or +1
        b = torch.randint_like(y, low=0, high=2).float()
        b = 2.0 * b - 1.0

        f_y_eps = model(y + eps * b)

        # Hutchinson finite-difference divergence estimate
        div = torch.sum(b * (f_y_eps - f_y)) / eps
        div_estimates.append(div)

    div_term = torch.stack(div_estimates).mean() / num_pixels

    sure_value = mse_term - sigma ** 2 + 2.0 * (sigma ** 2) * div_term

    return sure_value.item(), mse_term.item(), div_term.item()

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

def train_model_with_prior(model, input, path, learning_rate=1e-3, learning_rate_prior = 1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model = model.to(device) #reconstruction model
    input = input.to(device)
    z = input.clone().detach().requires_grad_(True) #prior
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": learning_rate},
                                  {"params": [z], "lr": learning_rate_prior}])
   
    model.train()

    loss_history = []

    os.makedirs(path, exist_ok=True)
                                 
    for it in range(num_iter):      
         
        masked_input, mask = random_patch_mask(
            z,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,         
            epoch=it            
        )

        output = model(masked_input)

        loss = F.mse_loss(output * (1 - mask), input * (1 - mask)) 
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
               plt.imshow(z.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               plt.show()
        model.train()
    np.save(os.path.join(path, 'loss_history.npy'), np.array(loss_history)) 

def train_model_SURE(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42):    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input = input.to(device)
    model.train()

    loss_history = []
    sure = []

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
        sure.append(sure_mc(model, input, 0.6)[2])

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
    plt.plot(sure)
  
    
    
   
