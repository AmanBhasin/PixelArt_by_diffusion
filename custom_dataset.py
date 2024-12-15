import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform=None, nums_sprites=2000, null_context=False):
        # Load the sprites and labels from the given file names
        self.sprites = np.load(sfilename)
        self.sprites = self.sprites[:nums_sprites]  # Limit the sprites to the specified number
        self.slabels = np.load(lfilename)
        self.slabels = self.slabels[:nums_sprites]  # Limit the labels to the specified number
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform  # Transform to apply to images
        self.null_context = null_context  # Whether to set labels to 0 (null context)
        self.sprites_shape = self.sprites.shape  # Store shapes for later use
        self.slabel_shape = self.slabels.shape

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.sprites)

    def __getitem__(self, idx):
        """Get the image and label at a given index."""
        image = self.sprites[idx]
        if self.transform:
            image = self.transform(image)
        
        # If null context is set, label is 0, otherwise fetch the actual label
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = torch.tensor(self.slabels[idx]).to(torch.int64)
        
        return (image, label)
