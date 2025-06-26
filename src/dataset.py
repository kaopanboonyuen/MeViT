# ------------------------------------------------------------------------------
# MeViT: Satellite Dataset Loading and Preprocessing for Semantic Segmentation
# Author: Teerapong Panboonyuen (Original)
# License: MIT License
#
# Description:
# This file defines a flexible custom dataset class for loading and preprocessing
# satellite imagery from multiple sources (e.g., Landsat-8, Sentinel-2, Google Earth)
# for semantic segmentation using the MeViT model.
# ------------------------------------------------------------------------------

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, input_size=(256, 256), image_exts=('.png', '.jpg', '.tif')):
        """
        Args:
            image_dir (str): Directory containing satellite images.
            label_dir (str): Directory containing corresponding label masks.
            input_size (tuple): Size to resize input images and labels.
            image_exts (tuple): Supported image extensions.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.image_exts = image_exts

        # Find all image files with supported extensions
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)]
        self.image_files.sort()  # Ensure consistent ordering

        self.image_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize(self.input_size, interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        # Load corresponding label (assuming same filename)
        label_path = os.path.join(self.label_dir, image_filename)
        label = Image.open(label_path).convert("L")  # Single-channel label mask

        # Apply transformations
        image = self.image_transform(image)
        label = self.label_transform(label)

        # Convert label to tensor manually (without normalization)
        label = transforms.functional.pil_to_tensor(label).squeeze(0).long()  # shape: [H, W]

        return image, label