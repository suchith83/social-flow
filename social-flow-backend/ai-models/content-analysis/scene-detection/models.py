"""
Scene Classification Models
"""

import torch
import torch.nn as nn
from torchvision import models
from .config import SceneConfig


class SceneClassifier(nn.Module):
    def __init__(self, num_classes=SceneConfig.NUM_CLASSES, model_type=SceneConfig.MODEL_TYPE):
        super().__init__()
        if model_type == "resnet18":
            backbone = models.resnet18(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
            self.model = backbone
        else:
            raise ValueError("Unsupported model type")

    def forward(self, x):
        return self.model(x)


def load_pretrained_model(path: str, device="cpu"):
    model = SceneClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
