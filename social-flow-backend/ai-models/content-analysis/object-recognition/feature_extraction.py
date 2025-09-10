"""
Feature Extraction for Object Recognition

Supports:
- CNN features (ResNet)
- Transformer features (ViT)
"""

import torch
from torchvision import models
from transformers import ViTModel, ViTFeatureExtractor
from .config import ObjectConfig


class FeatureExtractor:
    def __init__(self, model_type=ObjectConfig.MODEL_TYPE):
        self.model_type = model_type

        if model_type == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(backbone.children())[:-1])  # remove fc
            self.model.eval()
        elif model_type == "vit":
            self.extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.model.eval()
        else:
            raise ValueError("Unsupported model type")

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        if self.model_type == "resnet50":
            with torch.no_grad():
                feats = self.model(images)
            return feats.view(feats.size(0), -1)
        elif self.model_type == "vit":
            inputs = self.extractor(images, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.pooler_output
