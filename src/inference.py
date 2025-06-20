# ------------------------------------------------------------------------------
# MeViT: Inference Script
# Author: Teerapong Panboonyuen
# License: MIT License
# 
# Description:
# This script uses the trained MeViT model to perform inference on new data
# and generate segmentation maps.
# ------------------------------------------------------------------------------

import torch
from model import MeViT
from utils import load_config, load_model, process_image, save_segmentation_map

def inference(image_path, output_path):
    # Load configuration
    config = load_config("config.yaml")
    
    # Load trained model
    model = MeViT(config['model']['num_classes'])
    load_model(model, "mevit_model.pth")
    model.eval()
    
    # Process the input image
    image = process_image(image_path, config['model']['input_size'])
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
    
    # Save the segmentation map
    save_segmentation_map(pred, output_path)

if __name__ == "__main__":
    image_path = "/path/to/image.png"
    output_path = "/path/to/output_segmentation.png"
    inference(image_path, output_path)