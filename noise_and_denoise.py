#Include this function in the main training loop to visualize the diffusion progress for a given epoch using the cosine schedule.

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from torchvision.utils import save_image, make_grid
import os, math 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),                
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

image = Image.open("/path to image")
plt.imshow(image)

image_sample = transform(image)
print(image_sample.shape)

def visualize_diffusion_progress_cosine(epoch, image, timesteps, nn_model, ab_t, device="cuda"):
    """
    Visualizes the diffusion progress for a given epoch using the cosine schedule.
    
    Args:
        epoch (int): Current epoch number.
        image (torch.Tensor): Input image tensor.
        timesteps (int): Total number of timesteps in the diffusion process.
        nn_model (torch.nn.Module): Neural network predicting the noise.
        ab_t (torch.Tensor): Precomputed alpha_bar schedule (cumulative product).
    """
    nn_model.eval()

    original_image = image.to(device)
 
    t = torch.randint(1, timesteps + 1, (1,), device=device)

    noise = torch.randn_like(original_image).to(device)
    x_t = ab_t.sqrt()[t, None, None, None] * original_image + (1 - ab_t[t, None, None, None]).sqrt() * noise
    print(x_t.shape)

    with torch.no_grad():
        predicted_noise = nn_model(x_t, t / timesteps).squeeze(0)

    residual_noise = noise - predicted_noise

    denoised_image = (x_t - (1 - ab_t[t, None, None, None]).sqrt() * predicted_noise) / ab_t.sqrt()[t, None, None, None]

    to_numpy = lambda x: x.squeeze(0).detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() if x.dim() == 4 else x.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    
    original = to_numpy((original_image + 1) / 2)  
    noisy = to_numpy((x_t + 1) / 2)
    predicted = to_numpy((predicted_noise + 1) / 2)
    residual = to_numpy((residual_noise + 1) / 2)
    denoised = to_numpy((denoised_image + 1) / 2)

    plt.figure(figsize=(15, 5))
    titles = ["Original", "Noisy", "Predicted Noise", "Residual Noise", "Denoised"]
    images = [original, noisy, predicted, residual, denoised]
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.suptitle(f"Diffusion Visualization - Epoch {epoch}")

    save_dir = "diffusion_images_cosine"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()

    if epoch % 30 == 0:
        print(f"Displaying visualization for epoch {epoch}")
        plt.show()
    else:
        print(f"Visualization for epoch {epoch} saved at {save_path}")

    nn_model.train()
