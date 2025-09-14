# Implements database sharding logic
from typing import Dict, Any
import hashlib


class ShardManager:
    """
    Consistent Hashing Shard Manager.
    """

    def __init__(self, shards: Dict[str, Any]):
        self.shards = shards
        self.keys = list(shards.keys())

    def get_shard(self, key: str) -> str:
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return self.keys[h % len(self.keys)]
