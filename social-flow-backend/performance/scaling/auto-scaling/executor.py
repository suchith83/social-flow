# Executes scaling actions (scale up/down)
# performance/scaling/auto_scaling/executor.py

import asyncio
import logging
from .exceptions import ExecutionError


logger = logging.getLogger("auto_scaling.executor")


class Executor:
    """
    Executes scaling actions by interacting with the infrastructure layer.
    """

    def __init__(self):
        self.current_instances = 1

    async def scale_to(self, target_instances: int):
        """
        Scale cluster to target_instances asynchronously.
        """
        try:
            diff = target_instances - self.current_instances
            if diff > 0:
                await self._scale_up(diff)
            elif diff < 0:
                await self._scale_down(-diff)
            self.current_instances = target_instances
            logger.info(f"Cluster scaled to {self.current_instances} instances")
        except Exception as e:
            raise ExecutionError(f"Scaling execution failed: {e}")

    async def _scale_up(self, count: int):
        await asyncio.sleep(count * 0.5)
        logger.info(f"Scaled up by {count}")

    async def _scale_down(self, count: int):
        await asyncio.sleep(count * 0.5)
        logger.info(f"Scaled down by {count}")
