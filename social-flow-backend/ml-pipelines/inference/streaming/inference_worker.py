# Performs streaming inference
# ================================================================
# File: model_loader.py
# Purpose: Load ML models with versioning
# ================================================================

import logging
import joblib
import torch

logger = logging.getLogger("ModelLoader")


class ModelLoader:
    def __init__(self, config: dict):
        self.config = config

    def load_model(self):
        model_type = self.config.get("type", "sklearn")
        path = self.config["path"]

        if model_type == "torch":
            model = torch.load(path, map_location="cpu")
            model.eval()
            logger.info("✅ Torch model loaded")
            return model
        elif model_type == "sklearn":
            model = joblib.load(path)
            logger.info("✅ Sklearn model loaded")
            return model
        else:
            raise ValueError("Unsupported model type")

    def warmup(self, model):
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
