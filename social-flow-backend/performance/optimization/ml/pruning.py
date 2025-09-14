# Functions for pruning neural network parameters
import torch.nn.utils.prune as prune
from torch import nn


class Pruner:
    """
    Model pruning strategies:
    - Magnitude-based
    - Structured (filters, channels)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def magnitude_prune(self, amount: float = 0.3):
        """Global unstructured magnitude pruning."""
        parameters_to_prune = [
            (m, "weight")
            for m in self.model.modules()
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)
        ]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

    def structured_prune(self, module: nn.Module, amount: float = 0.5):
        """Prune filters/channels in Conv layers."""
        prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

    def remove_pruning(self):
        """Remove pruning reparameterizations permanently."""
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                prune.remove(m, "weight")
