# Routes queries to correct shard
# performance/scaling/sharding/dispatcher.py

import asyncio
import logging
from typing import Dict
from .exceptions import DispatchError


logger = logging.getLogger("sharding.dispatcher")


class Dispatcher:
    """
    Routes requests to the correct shard.
    """

    def __init__(self, shards: Dict[str, Dict]):
        self.shards = shards

    async def dispatch(self, shard: str, request: Dict):
        if shard not in self.shards:
            raise DispatchError(f"Shard {shard} not found")
        await asyncio.sleep(0.05)
        logger.info(f"Dispatched request {request['id']} with key={request['key']} to shard {shard}")
