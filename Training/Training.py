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

def train_model_self_guided(model, input, path, sigma, reg_coef, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42):    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input = input.to(device)

    input_1 = input
    input_2 = input
    
    model.train()

    loss_history = []

    os.makedirs(path, exist_ok=True)

    for it in range(num_iter):
        input_1 =  input + sigma * torch.randn_like(input)
        input_2 =  input + sigma * torch.randn_like(input)         

        
        masked_input_1, mask = random_patch_mask(
            input_1,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,         
            epoch=it            
        )

        masked_input_2, mask = random_patch_mask(
            input_2,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,         
            epoch=it            
        )

        output_1 = model(masked_input_1)
        output_2 = model(masked_input_2)

        loss = F.mse_loss(output_1 * (1 - mask), input * (1 - mask)) + reg_coef*torch.mean((output_1*(1-mask) - (output_2*(1-mask)).detach())**2)
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
    

        
   
