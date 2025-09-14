# Implements model quantization techniques
import torch
from torch import nn


class Quantizer:
    """
    Model quantization for performance optimization.
    Supports:
    - Dynamic quantization
    - Static quantization
    - Mixed precision
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def dynamic_quantize(self) -> nn.Module:
        """Apply dynamic quantization (works well for linear layers)."""
        return torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )

    def static_quantize(self, calibration_loader) -> nn.Module:
        """Apply static quantization with calibration dataset."""
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(self.model, inplace=True)
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                self.model(inputs)
        return torch.quantization.convert(self.model, inplace=True)

    def mixed_precision(self) -> nn.Module:
        """Use mixed precision with autocast (to be applied during training/inference)."""
        return self.model
