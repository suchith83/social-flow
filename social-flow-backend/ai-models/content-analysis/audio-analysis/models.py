"""
Models for Audio Analysis

Includes:
- CNN classifier
- RNN classifier
- Option to load pretrained models
"""

import torch
import torch.nn as nn
from .config import AudioConfig


class CNNBlock(nn.Module):
    """A simple CNN block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=AudioConfig.NUM_CLASSES):
        super().__init__()
        self.cnn1 = CNNBlock(1, 16)
        self.cnn2 = CNNBlock(16, 32)
        self.cnn3 = CNNBlock(32, 64)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)  # adjust dimensions based on input size
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(self.fc1(x))
        return self.fc2(x)


def load_pretrained_model(path: str, device="cpu"):
    """Load saved model"""
    model = AudioClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
