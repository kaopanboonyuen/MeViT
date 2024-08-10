# ------------------------------------------------------------------------------
# MeViT: Model Implementation
# Author: Teerapong Panboonyuen
# License: MIT License
# 
# Description:
# This file contains the implementation of the MeViT model, including the
# Medium-Resolution Vision Transformer and the revised MixCFN block.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixCFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixCFN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class MeViT(nn.Module):
    def __init__(self, num_classes):
        super(MeViT, self).__init__()
        self.conv_stem = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.mixcfn1 = MixCFN(64, 128)
        self.mixcfn2 = MixCFN(128, 256)
        self.mixcfn3 = MixCFN(256, 512)
        self.segmentation_head = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.mixcfn1(x)
        x = self.mixcfn2(x)
        x = self.mixcfn3(x)
        x = self.segmentation_head(x)
        return x