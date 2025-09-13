# Core real-time inference logic
# ================================================================
# File: inference_service.py
# Purpose: Core real-time inference logic
# ================================================================

import logging
import numpy as np
import torch
from typing import Any

logger = logging.getLogger("InferenceService")


class InferenceService:
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config

    async def infer(self, data: Any):
        """Perform inference with low latency"""
        try:
            if hasattr(self.model, "predict"):  # sklearn-like
                return self.model.predict([data]).tolist()
            elif isinstance(self.model, torch.nn.Module):
                X = torch.tensor(np.array([data]), dtype=torch.float32)
                with torch.no_grad():
                    return self.model(X).numpy().tolist()
            else:
                raise ValueError("Unsupported model type")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
