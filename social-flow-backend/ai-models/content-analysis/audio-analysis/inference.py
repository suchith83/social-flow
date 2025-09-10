"""
Inference Engine

Handles:
- Model loading
- Preprocessing
- Feature extraction
- Predictions
"""

import torch
import numpy as np
from .preprocessing import AudioPreprocessor
from .feature_extraction import FeatureExtractor
from .models import load_pretrained_model
from .config import AudioConfig


class InferenceEngine:
    def __init__(self, model_path: str, device="cpu"):
        self.device = device
        self.model = load_pretrained_model(model_path, device=device)
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()

    def predict(self, filepath: str) -> int:
        """Run inference on a single audio file"""
        y = self.preprocessor.preprocess(filepath)
        features = self.extractor.extract_mfcc(y)  # Using MFCC as input
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            pred = torch.argmax(logits, dim=1).item()
        return pred
