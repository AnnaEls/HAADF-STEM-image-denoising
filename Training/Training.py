import torch
import torch.nn.functional as F

from Training.Masking import random_patch_mask, random_validation_mask, random_patch_mask_with_validation
from Utilities.Utils import convert 

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import copy

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

def train_hybrid_model(model, input, path, learning_rate=1e-3, num_iter=1, patch_size=1, mask_ratio=0.2, show_image=False, seed=42):    
    
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
        model.train()
    np.save(os.path.join(path, 'loss_history.npy'), np.array(loss_history))    
 
     
def train_hybrid_model_with_validation(
    model,
    input,
    path,
    learning_rate=1e-3,
    num_iter=1000,
    patch_size=1,
    mask_ratio=0.2,
    val_ratio=0.05,
    patience=30,
    min_delta=1e-5,
    show_image=False,
    seed=42,
    val_seed=1234
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    input = input.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    os.makedirs(path, exist_ok=True)

    # ============================================================
    # FIXED VALIDATION PIXELS
    # ============================================================

    val_mask = random_validation_mask(
        input,
        val_ratio=val_ratio,
        seed=val_seed
    )

    # Validation input:
    # validation target pixels are always hidden
    val_visible_mask = (~val_mask).float()

    val_input = input * val_visible_mask

    # Save validation mask for inspection
    np.save(
        os.path.join(path, "validation_mask.npy"),
        val_mask.squeeze().cpu().numpy()
    )

    # ============================================================
    # HISTORIES
    # ============================================================

    loss_history = []
    val_loss_history = []

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    counter = 0

    # ============================================================
    # TRAINING
    # ============================================================

    for it in range(num_iter):

        model.train()

        # --------------------------------------------------------
        # Generate a NEW random training mask each epoch
        # --------------------------------------------------------

        masked_input, mask = random_patch_mask_with_validation(
            input,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            seed=seed,
            epoch=it
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # validation pixels must NOT be training targets
        # --------------------------------------------------------

        mask = mask.clone()

        # force validation pixels to remain visible
        # in the training input
        mask[val_mask] = 1.0

        masked_input = input * mask

        # Training target locations
        train_target_mask = (mask == 0)

        # Safety check:
        # no validation pixel may be a training target
        assert not torch.any(
            train_target_mask & val_mask
        )

        # --------------------------------------------------------
        # FORWARD
        # --------------------------------------------------------

        output_afno, output_cnn = model(
            masked_input
        )

        # --------------------------------------------------------
        # TRAINING LOSS
        #
        # Calculate loss ONLY at masked pixels
        # --------------------------------------------------------

        loss_afno = masked_mse(
            output_afno,
            input,
            train_target_mask
        )

        loss_cnn = masked_mse(
            output_cnn,
            input,
            train_target_mask
        )

        loss = loss_afno + loss_cnn

        # --------------------------------------------------------
        # OPTIMIZATION
        # --------------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_history.append(
            loss.item()
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        model.eval()

        with torch.no_grad():

            val_output_afno, val_output_cnn = model(
                val_input
            )

            val_loss_afno = masked_mse(
                val_output_afno,
                input,
                val_mask
            )

            val_loss_cnn = masked_mse(
                val_output_cnn,
                input,
                val_mask
            )

            val_loss = (
                val_loss_afno +
                val_loss_cnn
            )

        val_loss_history.append(
            val_loss.item()
        )

        # ========================================================
        # CHECKPOINT / EARLY STOPPING
        # ========================================================

        if val_loss.item() < best_val_loss - min_delta:

            best_val_loss = val_loss.item()

            best_epoch = it

            counter = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_state,
                os.path.join(
                    path,
                    "best_model.pt"
                )
            )

        else:

            counter += 1

        # ========================================================
        # FULL IMAGE RECONSTRUCTION
        # ========================================================

        with torch.no_grad():

            denoised_image_afno, denoised_image_cnn = model(
                input
            )

        # --------------------------------------------------------
        # SAVE
        # --------------------------------------------------------

        tifffile.imwrite(
            f'{path}/AFNO_{(it+1):04d}.tif',
            convert(
                denoised_image_afno
                .squeeze()
                .cpu()
                .numpy()
            ),
            imagej=True
        )

        tifffile.imwrite(
            f'{path}/CNN_{(it+1):04d}.tif',
            convert(
                denoised_image_cnn
                .squeeze()
                .cpu()
                .numpy()
            ),
            imagej=True
        )

        # --------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------

        if show_image:

            print(
                f"epoch {it+1:4d} | "
                f"train={loss.item():.6f} | "
                f"val={val_loss.item():.6f} | "
                f"best={best_val_loss:.6f} "
                f"(epoch {best_epoch+1})"
            )

            plt.figure(figsize=(8, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(
                denoised_image_afno
                .squeeze()
                .cpu()
                .numpy(),
                cmap="gray"
            )
            plt.title("AFNO")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(
                denoised_image_cnn
                .squeeze()
                .cpu()
                .numpy(),
                cmap="gray"
            )
            plt.title("CNN")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(
                val_mask
                .squeeze()
                .cpu()
                .numpy(),
                cmap="gray"
            )
            plt.title("Validation pixels")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        # ========================================================
        # EARLY STOP
        # ========================================================

        if counter >= patience:

            print(
                f"\nEarly stopping at epoch {it+1}"
            )

            print(
                f"Best epoch: {best_epoch+1}"
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.6f}"
            )

            break

    # ============================================================
    # RESTORE BEST MODEL
    # ============================================================

    if best_state is not None:
        model.load_state_dict(best_state)

    # ============================================================
    # SAVE LOSS HISTORIES
    # ============================================================

    np.save(
        os.path.join(
            path,
            "loss_history.npy"
        ),
        np.array(loss_history)
    )

    np.save(
        os.path.join(
            path,
            "val_loss_history.npy"
        ),
        np.array(val_loss_history)
    )

    # ============================================================
    # SAVE BEST RECONSTRUCTION
    # ============================================================

    model.eval()

    with torch.no_grad():

        best_afno, best_cnn = model(input)

    tifffile.imwrite(
        os.path.join(
            path,
            "AFNO_best.tif"
        ),
        convert(
            best_afno
            .squeeze()
            .cpu()
            .numpy()
        ),
        imagej=True
    )

    tifffile.imwrite(
        os.path.join(
            path,
            "CNN_best.tif"
        ),
        convert(
            best_cnn
            .squeeze()
            .cpu()
            .numpy()
        ),
        imagej=True
    )

    print(
        f"\nBest epoch = {best_epoch+1}"
    )

    print(
        f"Best validation loss = "
        f"{best_val_loss:.6f}"
    )

    return (
        model,
        np.array(loss_history),
        np.array(val_loss_history),
        best_epoch
    )  

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
