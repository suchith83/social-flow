# Orchestrates sharding + rebalancing
# performance/scaling/sharding/orchestrator.py

import asyncio
import logging
from typing import Dict, Any

from .algorithms import HashSharding, RangeSharding, ConsistentHashing
from .dispatcher import Dispatcher
from .monitor import Monitor
from .rebalancer import Rebalancer
from .exceptions import ShardingError


logger = logging.getLogger("sharding.orchestrator")


class Orchestrator:
    """
    Orchestrates sharding operations: routing, monitoring, and rebalancing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shards = config["shards"]
        self.monitor = Monitor(self.shards)
        self.dispatcher = Dispatcher(self.shards)
        self.rebalancer = Rebalancer(self.shards)

        algo = config["algorithm"]
        if algo == "hash":
            self.algo = HashSharding(self.shards)
        elif algo == "range":
            self.algo = RangeSharding(self.shards)
        elif algo == "consistent_hash":
            self.algo = ConsistentHashing(self.shards)
        else:
            raise ValueError(f"Unknown sharding algorithm: {algo}")

    async def handle_request(self, request: Dict):
        try:
            shard = self.algo.select(request["key"])
            await self.dispatcher.dispatch(shard, request)
            await self.monitor.update_load(shard, +1)
        except Exception as e:
            raise ShardingError(f"Request handling failed: {e}")

    async def start(self):
        asyncio.create_task(self.monitor.simulate_load())
        req_id = 0
        while True:
            req_id += 1
            request = {"id": req_id, "key": str(req_id * 7)}
            await self.handle_request(request)
            await asyncio.sleep(1.0)
