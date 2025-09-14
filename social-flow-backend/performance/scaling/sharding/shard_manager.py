# Maintains shard metadata
# performance/scaling/sharding/shard_manager.py

import logging
from typing import Dict


logger = logging.getLogger("sharding.shard_manager")


class ShardManager:
    """
    Maintains shard metadata and provides API to add/remove shards.
    """

    def __init__(self, shards: Dict[str, Dict]):
        self.shards = shards

    def add_shard(self, name: str, node: str, range_: tuple = None):
        self.shards[name] = {"node": node}
        if range_:
            self.shards[name]["range"] = range_
        logger.info(f"Added shard {name} at {node}")

    def remove_shard(self, name: str):
        if name in self.shards:
            del self.shards[name]
            logger.info(f"Removed shard {name}")

    def get_shards(self) -> Dict[str, Dict]:
        return self.shards
