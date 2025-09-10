"""
Feature Extraction for Scene Detection

Uses CNN features for frame embeddings.
"""

import torch
from torchvision import models
from .config import SceneConfig


class FeatureExtractor:
    def __init__(self, model_type=SceneConfig.MODEL_TYPE):
        if model_type == "resnet18":
            backbone = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        elif model_type == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError("Unsupported model type")
        self.model.eval()

    def extract(self, frames):
        """Extract features for frames (batch of tensors)"""
        with torch.no_grad():
            feats = [self.model(f.unsqueeze(0)).view(-1) for f in frames]
        return torch.stack(feats)
