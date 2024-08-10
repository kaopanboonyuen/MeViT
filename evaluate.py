# ------------------------------------------------------------------------------
# MeViT: Evaluation Script
# Author: Teerapong Panboonyuen
# License: MIT License
# 
# Description:
# This script evaluates the MeViT model using the validation dataset and calculates
# metrics such as precision, recall, F1 score, and mean IoU.
# ------------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from model import MeViT
from dataset import CustomDataset
from utils import load_config, load_model, calculate_metrics

def evaluate():
    # Load configuration
    config = load_config("config.yaml")
    
    # Initialize dataset and data loader
    val_dataset = CustomDataset(config['dataset']['val_path'], config['model']['input_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Load trained model
    model = MeViT(config['model']['num_classes'])
    load_model(model, "mevit_model.pth")
    model.eval()
    
    # Evaluate the model
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, config['evaluation']['metrics'])
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    evaluate()