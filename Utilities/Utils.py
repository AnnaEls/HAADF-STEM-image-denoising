import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt

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
