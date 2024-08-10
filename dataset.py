# ------------------------------------------------------------------------------
# MeViT: Dataset Loading and Preprocessing
# Author: Teerapong Panboonyuen
# License: MIT License
# 
# Description:
# This file contains the custom dataset class used to load and preprocess the
# Landsat satellite imagery for training and evaluation.
# ------------------------------------------------------------------------------

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_path, input_size):
        self.data_path = data_path
        self.image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(image_path.replace('images', 'labels')).convert("L")
        image = self.transform(image)
        label = self.transform(label)
        return image, label