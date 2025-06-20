# ------------------------------------------------------------------------------
# MeViT++: Advanced Medium-Resolution Vision Transformer
# Author: Teerapong Panboonyuen (Enhanced by AI Collaboration)
# License: MIT License
#
# Description:
# A mathematically enriched version of MeViT. Introduces spectral normalization,
# channel-wise attention, hierarchical feature fusion, and enhanced spatial modeling
# to further improve semantic segmentation performance on Landsat imagery.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding Layer for Convolutional Feature Maps
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.pe = self.create_positional_encoding()

    def create_positional_encoding(self):
        pe = torch.zeros(1, self.channels, self.height, self.width)
        for y in range(self.height):
            for x in range(self.width):
                for c in range(0, self.channels, 2):
                    div_term = math.exp(c * -math.log(10000.0) / self.channels)
                    pe[0, c, y, x] = math.sin(x * div_term)
                    if c + 1 < self.channels:
                        pe[0, c+1, y, x] = math.cos(y * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return x + self.pe.to(x.device)

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se.expand_as(x)

# Enhanced MixCFN Block with Multi-Scale Features and Spectral Norm
class MixCFNPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixCFNPlus, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv5x5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels)
        self.pointwise = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(0.15)
        self.relu = nn.GELU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.pointwise(x)
        x = self.se(x)
        x = self.norm(self.dropout(self.relu(x)))
        return x + residual  # Residual connection

# Gaussian Noise Injection (used for data regularization)
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

# MeViT++ Model
class MeViTPlus(nn.Module):
    def __init__(self, num_classes, input_size=(224, 224)):
        super(MeViTPlus, self).__init__()
        H, W = input_size
        self.noise = GaussianNoise(std=0.05)
        self.pos_enc = PositionalEncoding2D(64, H//2, W//2)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = MixCFNPlus(64, 128)
        self.layer2 = MixCFNPlus(128, 256)
        self.layer3 = MixCFNPlus(256, 512)

        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 256 + 128, 512, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(512)
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.noise(x)
        x = self.conv_stem(x)
        x = self.pos_enc(x)

        x1 = self.layer1(x)  # 128 channels
        x2 = self.layer2(x1) # 256 channels
        x3 = self.layer3(x2) # 512 channels

        # Resize all to same spatial shape for fusion
        x1_up = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x2_up = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([x1_up, x2_up, x3], dim=1))

        out = self.segmentation_head(fused)
        return out