# Utilities for model/data parallelism
import torch
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


class ParallelTrainer:
    """
    Parallelism strategies:
    - DataParallel (single-node multi-GPU)
    - DistributedDataParallel (multi-node)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def data_parallel(self) -> nn.Module:
        """Wrap model in DataParallel."""
        return DataParallel(self.model)

    def distributed(self, device_ids=None) -> nn.Module:
        """Wrap model in DistributedDataParallel."""
        return DistributedDataParallel(self.model, device_ids=device_ids)
