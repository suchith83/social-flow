# Shard-level monitoring
# performance/scaling/sharding/monitor.py

import asyncio
import logging
import random
from typing import Dict


logger = logging.getLogger("sharding.monitor")


class Monitor:
    """
    Monitors shard-level load.
    """

    def __init__(self, shards: Dict[str, Dict]):
        self.shards = {s: {"load": 0, **d} for s, d in shards.items()}

    async def update_load(self, shard: str, delta: int):
        if shard in self.shards:
            self.shards[shard]["load"] += delta
            logger.debug(f"Shard {shard} load updated: {self.shards[shard]['load']}")

    async def simulate_load(self):
        while True:
            for shard in self.shards:
                self.shards[shard]["load"] = random.randint(0, 100)
            logger.info(f"Shard loads: { {s: d['load'] for s, d in self.shards.items()} }")
            await asyncio.sleep(5)
