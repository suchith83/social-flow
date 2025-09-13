# Loads models with warmup
# ================================================================
# File: model_loader.py
# Purpose: Load and warm ML models for low-latency inference
# ================================================================

import logging
import joblib
import torch
import time

logger = logging.getLogger("ModelLoader")


class ModelLoader:
    def __init__(self, config: dict):
        self.config = config

    def load_model(self):
        path = self.config["path"]
        model_type = self.config.get("type", "sklearn")

        if model_type == "torch":
            model = torch.load(path, map_location="cpu")
            model.eval()
            logger.info(f"✅ Torch model loaded: {path}")
            return model
        elif model_type == "sklearn":
            model = joblib.load(path)
            logger.info(f"✅ Sklearn model loaded: {path}")
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def warmup(self, model):
        """Run a dummy inference to allocate memory / JIT compile"""
        try:
            if isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    dummy_input = torch.zeros((1, self.config.get("input_dim", 10)))
                    _ = model(dummy_input)
            else:
                _ = model.predict([[0] * self.config.get("input_dim", 10)])
            logger.info("🔥 Model warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
