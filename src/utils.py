# ------------------------------------------------------------------------------
# MeViT: Utility Functions
# Author: Teerapong Panboonyuen (Original), Revised by ChatGPT
# License: MIT License
#
# Description:
# Utility functions for the MeViT project, including configuration management,
# model checkpointing, image preprocessing, segmentation output saving, and
# comprehensive semantic segmentation metrics calculation.
# ------------------------------------------------------------------------------

import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score

def load_config(config_path: str):
    """
    Load YAML configuration from file.

    Args:
        config_path (str): Path to YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_model(model: torch.nn.Module, path: str):
    """
    Save PyTorch model weights to disk.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Destination file path.
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: torch.device = None):
    """
    Load model weights from disk into the provided model instance.

    Args:
        model (torch.nn.Module): Model instance to load weights into.
        path (str): Path to saved weights file.
        device (torch.device, optional): Device to map weights to (CPU/GPU). Defaults to None.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    map_location = device if device else 'cpu'
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


def preprocess_image(image_path: str, input_size: tuple):
    """
    Load and preprocess an image for model input.

    Args:
        image_path (str): Path to the input image.
        input_size (tuple): Desired (H, W) size.

    Returns:
        torch.Tensor: Preprocessed image tensor, shape (1, 3, H, W).
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def save_segmentation_map(pred_mask: torch.Tensor, output_path: str):
    """
    Save a predicted segmentation mask as an image.

    Args:
        pred_mask (torch.Tensor): Prediction tensor (H, W) or (1, H, W).
        output_path (str): File path to save the image.
    """
    if pred_mask.dim() == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)

    pred_np = pred_mask.cpu().numpy().astype(np.uint8)
    Image.fromarray(pred_np).save(output_path)


def calculate_segmentation_metrics(labels, preds, metrics=('precision', 'recall', 'f1_score', 'mean_iou', 'accuracy')):
    """
    Calculate semantic segmentation metrics.

    Args:
        labels (array-like): Ground truth labels, flattened.
        preds (array-like): Predicted labels, flattened.
        metrics (tuple): List of metrics to calculate.

    Returns:
        dict: Dictionary mapping metric names to values.
    """
    labels = np.array(labels).flatten()
    preds = np.array(preds).flatten()

    results = {}
    if "precision" in metrics:
        results["precision"] = precision_score(labels, preds, average='macro', zero_division=0)
    if "recall" in metrics:
        results["recall"] = recall_score(labels, preds, average='macro', zero_division=0)
    if "f1_score" in metrics:
        results["f1_score"] = f1_score(labels, preds, average='macro', zero_division=0)
    if "mean_iou" in metrics:
        results["mean_iou"] = jaccard_score(labels, preds, average='macro', zero_division=0)
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(labels, preds)

    return results