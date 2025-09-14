# Monitoring resource utilization & system metrics
# performance/scaling/auto_scaling/monitor.py

import psutil
import asyncio
import logging
from typing import Dict, Any


logger = logging.getLogger("auto_scaling.monitor")


class Monitor:
    """
    Collects system metrics like CPU, memory, and network usage.

    This class abstracts metric collection to feed scaling policies.
    """

    async def collect_metrics(self) -> Dict[str, Any]:
        """
        Collects current system metrics asynchronously.
        """
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        net = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

        metrics = {"cpu": cpu, "memory": memory, "network": net}
        logger.debug(f"Collected metrics: {metrics}")
        return metrics

    async def stream_metrics(self, interval: float = 5.0):
        """
        Asynchronously yields metrics every `interval` seconds.
        """
        while True:
            metrics = await self.collect_metrics()
            yield metrics
            await asyncio.sleep(interval)
