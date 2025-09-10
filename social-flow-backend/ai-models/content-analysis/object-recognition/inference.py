"""
Inference for Object Recognition
"""

import torch
from .preprocessing import ImagePreprocessor
from .models import load_pretrained_model


class InferenceEngine:
    def __init__(self, model_path: str, device="cpu"):
        self.device = device
        self.model = load_pretrained_model(model_path, device=device)
        self.preprocessor = ImagePreprocessor()

    def predict(self, filepath: str) -> int:
        """Predict class ID"""
        img = self.preprocessor.load_image(filepath).to(self.device)
        with torch.no_grad():
            logits = self.model(img)
            pred = torch.argmax(logits, dim=1).item()
        return pred
