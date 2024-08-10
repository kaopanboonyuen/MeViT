# ------------------------------------------------------------------------------
# MeViT: Utility Functions
# Author: Teerapong Panboonyuen
# License: MIT License
# 
# Description:
# This file contains utility functions for the MeViT project, including loading
# configurations, saving/loading models, processing images, and calculating metrics.
# ------------------------------------------------------------------------------

import yaml
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def process_image(image_path, input_size):
    from PIL import Image
    from torchvision.transforms import transforms
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def save_segmentation_map(pred, output_path):
    pred = pred.squeeze(0).cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    Image.fromarray(pred).save(output_path)

def calculate_metrics(labels, preds, metric_list):
    metrics = {}
    labels = np.array(labels)
    preds = np.array(preds)
    if "precision" in metric_list:
        metrics["precision"] = precision_score(labels, preds, average='macro')
    if "recall" in metric_list:
        metrics["recall"] = recall_score(labels, preds, average='macro')
    if "f1_score" in metric_list:
        metrics["f1_score"] = f1_score(labels, preds, average='macro')
    if "mean_iou" in metric_list:
        metrics["mean_iou"] = jaccard_score(labels, preds, average='macro')
    return metrics