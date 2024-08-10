# ------------------------------------------------------------------------------
# MeViT: Training Script
# Author: Teerapong Panboonyuen
# License: MIT License
# 
# Description:
# This script trains the MeViT model using the specified dataset and parameters
# from the config file.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MeViT
from dataset import CustomDataset
from utils import load_config, save_model

def train():
    # Load configuration
    config = load_config("config.yaml")
    
    # Initialize dataset and data loader
    train_dataset = CustomDataset(config['dataset']['train_path'], config['model']['input_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = MeViT(config['model']['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {running_loss/len(train_loader)}")
    
    # Save the trained model
    save_model(model, "mevit_model.pth")

if __name__ == "__main__":
    train()