#Run the below script to fine-tune the DDPM model on the pixel art dataset. Be sure to change the input and save paths to the correct paths.

import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMPipeline
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle 

image_dir = "/kaggle/input/pixel-with-bg/pixilatedDataset_bg"
save_dir = "/kaggle/working/"

class PixelArtDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False, max_samples=None):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)

        if(max_samples):
            self.sprites = self.sprites[:max_samples]
            self.slabels = self.slabels[:max_samples]
            
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape
  
    def __len__(self):
        return len(self.sprites)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.sprites[idx].astype(np.uint8))
        
        if self.transform:
            #image = self.transform(self.sprites[idx])
            image = self.transform(image)
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        return self.sprites_shape, self.slabel_shape

image_size = 32  

#palette = np.load("palette.npy")

def quantize_image(image, palette):
    pixels = np.array(image).reshape(-1, 3)

    distances = np.linalg.norm(pixels[:, None] - palette, axis=2)
    nearest_colors = palette[np.argmin(distances, axis=1)]
    quantized_pixels = nearest_colors.reshape(image.size[1], image.size[0], 3)
    quantized_tensor = torch.tensor(quantized_pixels, dtype=torch.float32).permute(2, 0, 1) / 255.0

    return quantized_tensor

custom_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),  
    transforms.ToTensor(),                       
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # to include colour quantization uncomment the below line
    #transforms.Lambda(quantize_image),
])

train_dataset = PixelArtDataset(sfilename="/kaggle/input/pixel-art/sprites.npy",lfilename= "/kaggle/input/pixel-art/sprites_labels.npy", transform=custom_transforms, max_samples=15000)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "google/ddpm-cifar10-32"
ddpm = DDPMPipeline.from_pretrained(model_id)
#pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
#pipe = pipe.to("cuda")
pipe = ddpm.to("cuda")
unet = pipe.unet 
noise_scheduler = pipe.scheduler

img_path = "/kaggle/input/pixel-32x32/images/o_hs_001_2.png"
img = Image.open(img_path)
plt.imshow(img)

def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

optimizer = optim.Adam(unet.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

num_epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

unet.to(device)

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        images = images.to(device)
        batch_size = images.size(0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(images).to(device)

        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        with torch.cuda.amp.autocast():  # Mixed precision for memory efficiency
            noise_pred = unet(noisy_images, timesteps)["sample"]

        # MSE Loss
        loss = loss_fn(noise_pred.float(), noise.float())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    if ((epoch+1)%10) == 0:
        #Saving model for every 10 epochs. Comment it out if not needed.
        checkpoint_path = os.path.join(save_dir, f"v5_epoch_{epoch + 1}.pth")
        save_checkpoint(epoch + 1, unet, optimizer, epoch_loss / len(train_loader), checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

model_id = "google/ddpm-cifar10-32"
checkpoint_path = "/kaggle/input/v5_340_ddpm/pytorch/default/1/v5_epoch_340.pth"
print(os.path.exists(checkpoint_path))

pipe = DDPMPipeline.from_pretrained(model_id).to("cuda")

checkpoint = torch.load(checkpoint_path, map_location="cuda", pickle_module=pickle)
pipe.unet.load_state_dict(checkpoint["model_state_dict"])

num_samples = 4

noise = torch.randn((num_samples, 3, 32, 32)).to("cuda")  # 32x32 image size

generated_images = pipe(
    batch_size=num_samples,          
    #generator=torch.manual_seed(0),  
    num_inference_steps=1000           
).images

def show_images(images):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))  
    for i, ax in enumerate(axes.flat):  
        ax.imshow(images[i])            
        ax.axis("off")                  
    plt.tight_layout()
    plt.show()

show_images(generated_images)
