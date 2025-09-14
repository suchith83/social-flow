# Caching strategies for ML workflows
import torch
import hashlib


class InferenceCache:
    """
    Inference caching:
    - Caches model outputs for identical inputs
    - Uses tensor hashing for keys
    """

    def __init__(self):
        self.cache = {}

    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

    def get(self, tensor: torch.Tensor):
        key = self._tensor_hash(tensor)
        return self.cache.get(key)

    def set(self, tensor: torch.Tensor, output: torch.Tensor):
        key = self._tensor_hash(tensor)
        self.cache[key] = output
