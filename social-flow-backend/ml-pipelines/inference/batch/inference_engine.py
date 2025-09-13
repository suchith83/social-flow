# Runs inference with optimizations
# ================================================================
# File: inference_engine.py
# Purpose: Run optimized batch inference
# ================================================================

import logging
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("InferenceEngine")


class InferenceEngine:
    """
    Supports batch inference with:
    - Threaded execution
    - Torch tensor acceleration
    - Numpy vectorization
    """

    def __init__(self, config: dict):
        self.batch_size = config.get("batch_size", 128)
        self.parallelism = config.get("parallelism", 4)

    def run(self, model, data):
        if hasattr(model, "predict"):  # sklearn-like API
            return model.predict(data)
        elif isinstance(model, torch.nn.Module):
            return self._run_torch(model, data)
        else:
            raise ValueError("Unsupported model type")

    def _run_torch(self, model, data):
        X = torch.tensor(np.array(data), dtype=torch.float32)
        results = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i+self.batch_size]
                outputs = model(batch)
                results.append(outputs.numpy())
        return np.vstack(results)
