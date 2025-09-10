"""
Inference for Scene Detection
"""

import torch
from .preprocessing import VideoPreprocessor
from .models import load_pretrained_model


class InferenceEngine:
    def __init__(self, model_path: str, device="cpu"):
        self.device = device
        self.model = load_pretrained_model(model_path, device=device)
        self.preprocessor = VideoPreprocessor()

    def classify_frames(self, video_path: str):
        frames = self.preprocessor.extract_frames(video_path)
        preds = []
        with torch.no_grad():
            for f in frames:
                f = f.unsqueeze(0).to(self.device)
                logits = self.model(f)
                pred = torch.argmax(logits, dim=1).item()
                preds.append(pred)
        return preds
