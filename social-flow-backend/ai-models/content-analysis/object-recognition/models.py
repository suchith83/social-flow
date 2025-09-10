"""
Object Recognition Models
"""

import torch
import torch.nn as nn
from torchvision import models
from .config import ObjectConfig


class ObjectRecognizer(nn.Module):
    def __init__(self, num_classes=ObjectConfig.NUM_CLASSES, model_type=ObjectConfig.MODEL_TYPE):
        super().__init__()
        if model_type == "resnet50":
            backbone = models.resnet50(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
            self.model = backbone
        elif model_type == "vit":
            from transformers import ViTForImageClassification
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=num_classes
            )
        else:
            raise ValueError("Unsupported model type")

    def forward(self, x):
        return self.model(x)


def load_pretrained_model(path: str, device="cpu"):
    model = ObjectRecognizer()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
