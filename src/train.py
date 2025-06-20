# ------------------------------------------------------------------------------
# MeViT: Training Script
# Author: Teerapong Panboonyuen
# License: MIT License
#
# Description:
# This script trains the MeViT model using configurations defined in 'config.yaml'.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MeViT
from dataset import CustomDataset
from utils import load_config, save_model

def train():
    # Load configuration from file
    config = load_config("config.yaml")
    
    # Initialize dataset and data loader
    train_dataset = CustomDataset(
        data_path=config['dataset']['train_path'],
        input_size=config['model']['input_size']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Initialize model, loss function, and optimizer
    model = MeViT(num_classes=config['model']['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    num_epochs = config['training']['epochs']
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
    
    # Save the final trained model
    save_model(model, "mevit_model.pth")
    print("✅ Model training complete and saved to 'mevit_model.pth'.")

if __name__ == "__main__":
    train()