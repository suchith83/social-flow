# Handles batching of data and operations
import asyncio
import torch
from typing import List


class DynamicBatcher:
    """
    Dynamic batching for ML inference.
    Collects requests within a time window and processes them together.
    """

    def __init__(self, model, batch_size: int = 32, timeout: float = 0.01):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue: List[torch.Tensor] = []
        self.lock = asyncio.Lock()

    async def infer(self, tensor: torch.Tensor):
        async with self.lock:
            self.queue.append(tensor)
            if len(self.queue) >= self.batch_size:
                return await self._flush()
        await asyncio.sleep(self.timeout)
        return await self._flush()

    async def _flush(self):
        if not self.queue:
            return None
        batch = torch.stack(self.queue)
        self.queue.clear()
        with torch.no_grad():
            return self.model(batch)
