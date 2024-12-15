from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.cluster import KMeans
import torch
# Please load the dataset in the data_dir
image_paths = glob.glob("/data_dir/Type Dataset/water/*.png")

colors = []

for path in image_paths:
    image = Image.open(path).convert("RGB")
    colors.extend(np.array(image).reshape(-1, 3))  # Flatten RGB values

colors = np.array(colors)

kmeans = KMeans(n_clusters=16, random_state=0)
kmeans.fit(colors)

# Extract the palette (cluster centers)
palette = kmeans.cluster_centers_.astype(int)

def quantize_image(image, palette):
    pixels = np.array(image).reshape(-1, 3)

    distances = np.linalg.norm(pixels[:, None] - palette, axis=2)
    nearest_colors = palette[np.argmin(distances, axis=1)]
    quantized_pixels = nearest_colors.reshape(image.size[1], image.size[0], 3)
    quantized_tensor = torch.tensor(quantized_pixels, dtype=torch.float32).permute(2, 0, 1) / 255.0

    return quantized_tensor
    #return Image.fromarray(quantized_pixels.astype('uint8'), "RGB")
    
quantized_image = quantize_image(Image.open("/data_dir/Type Dataset Gold/bug/012.png"), palette)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# Plot the second image
axes[1].imshow(quantized_image, cmap='viridis')
axes[1].set_title("Image after quantization")
axes[1].axis('off')

# Show the plot
plt.tight_layout()
plt.show()