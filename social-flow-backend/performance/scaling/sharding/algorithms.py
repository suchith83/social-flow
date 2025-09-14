# Sharding algorithms
# performance/scaling/sharding/algorithms.py

import hashlib
import logging
from typing import Dict, Any
from .exceptions import AlgorithmError


logger = logging.getLogger("sharding.algorithms")


class HashSharding:
    """
    Shard selection using hash-based partitioning.
    """

    def __init__(self, shards: Dict[str, Dict]):
        self.shards = shards

    def select(self, key: str) -> str:
        if not self.shards:
            raise AlgorithmError("No shards available")
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        shard_names = list(self.shards.keys())
        return shard_names[h % len(shard_names)]


class RangeSharding:
    """
    Shard selection using numeric ranges.
    """

    def __init__(self, shards: Dict[str, Dict]):
        self.shards = shards

    def select(self, key: int) -> str:
        for name, data in self.shards.items():
            low, high = data["range"]
            if low <= key <= high:
                return name
        raise AlgorithmError(f"No shard found for key {key}")


class ConsistentHashing:
    """
    Consistent hashing-based sharding.
    """

    def __init__(self, shards: Dict[str, Dict], replicas: int = 100):
        self.ring = {}
        self.sorted_keys = []
        for shard in shards.keys():
            for i in range(replicas):
                h = self._hash(f"{shard}:{i}")
                self.ring[h] = shard
                self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def select(self, key: str) -> str:
        if not self.ring:
            raise AlgorithmError("No shards in ring")
        h = self._hash(key)
        for k in self.sorted_keys:
            if h <= k:
                return self.ring[k]
        return self.ring[self.sorted_keys[0]]
