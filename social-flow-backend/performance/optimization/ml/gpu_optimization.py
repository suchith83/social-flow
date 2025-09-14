# GPU acceleration and optimization helpers
import torch


class GPUOptimizer:
    """
    GPU optimization strategies:
    - Device placement
    - Mixed precision inference
    """

    def __init__(self, model):
        self.model = model

    def to_device(self, device: str = "cuda"):
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        return self.model

    def mixed_precision_infer(self, inputs):
        """Perform inference with mixed precision (AMP)."""
        self.model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                return self.model(inputs)
