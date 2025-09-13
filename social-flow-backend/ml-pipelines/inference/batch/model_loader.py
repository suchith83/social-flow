# Handles model versioning and loading
# ================================================================
# File: model_loader.py
# Purpose: Loads model artifacts with versioning support
# ================================================================

import logging
import joblib
import torch
from utils import retry

logger = logging.getLogger("ModelLoader")


class ModelLoader:
    """
    Loads machine learning models from:
    - Local filesystem
    - Torch models
    - Joblib/sklearn
    - Cloud artifact store
    """

    def __init__(self, config: dict):
        self.config = config

    @retry(max_attempts=3, delay=5, exceptions=(Exception,))
    def load_model(self):
        model_type = self.config.get("type", "sklearn")
        path = self.config["path"]

        if model_type == "torch":
            model = torch.load(path, map_location="cpu")
            model.eval()
            return model
        elif model_type == "sklearn":
            return joblib.load(path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
