# Monitor region-level health & latency
# performance/scaling/regional/monitor.py

import asyncio
import random
import logging
from typing import Dict


logger = logging.getLogger("regional.monitor")


class Monitor:
    """
    Monitors latency and availability of regions.
    """

    def __init__(self, regions: Dict[str, Dict]):
        self.regions = {
            name: {"healthy": True, "latency": float("inf"), **data}
            for name, data in regions.items()
        }

    async def measure_latency(self, region: str) -> float:
        """
        Fake latency measurement (simulate with randomness).
        """
        await asyncio.sleep(0.1)
        latency = random.uniform(50, 300)  # ms
        self.regions[region]["latency"] = latency
        return latency

    async def update_metrics(self):
        tasks = [self.measure_latency(region) for region in self.regions.keys()]
        await asyncio.gather(*tasks)
        logger.debug(f"Updated latencies: { {r: d['latency'] for r, d in self.regions.items()} }")

    async def get_snapshot(self) -> Dict[str, Dict]:
        return self.regions
