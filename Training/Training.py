import torch
import torch.nn.functional as F

from Training.Masking import random_patch_mask
from Utilities.Utils import convert 

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

def z_score_normalize(x, eps=1e-8):
    mean = x.mean()
    std = x.std(unbiased=False)  # stable for single image
    x_norm = (x - mean) / (std + eps)
    return x_norm

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

def train_model_with_prior(model, input, path, learning_rate=1e-3, learning_rate_prior = 1e-3, sigma=0.1, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42):
    
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
        eps = sigma * torch.randn_like(z)
        input_eps = z_score_normalize(input + eps)
        masked_input, mask = random_patch_mask(
            z,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,         
            epoch=it            
        )

        output = model(masked_input)

        loss = F.mse_loss(output * (1 - mask), input_eps * (1 - mask))
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

    
    
   
