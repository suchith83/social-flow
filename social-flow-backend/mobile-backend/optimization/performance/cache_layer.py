# Multi-tier caching system
"""
Multi-Tier Cache Layer
Combines in-memory and persistent caching for performance.
"""

import time
from collections import OrderedDict
from .config import CONFIG


class CacheLayer:
    def __init__(self):
        self.memory_cache = OrderedDict()
        self.persistent_cache = {}

    def _evict_if_needed(self):
        """Evict old items if memory cache exceeds size."""
        while len(self.memory_cache) > CONFIG.memory_cache_size_mb * 1000:
            self.memory_cache.popitem(last=False)

    def get(self, key: str):
        """Retrieve from memory or persistent cache."""
        if key in self.memory_cache:
            return self.memory_cache[key]
        elif key in self.persistent_cache:
            val, expiry = self.persistent_cache[key]
            if expiry > time.time():
                return val
        return None

    def set(self, key: str, value: any, persistent: bool = False):
        """Store in memory or persistent cache."""
        self.memory_cache[key] = value
        self.memory_cache.move_to_end(key)
        self._evict_if_needed()

        if persistent:
            self.persistent_cache[key] = (value, time.time() + CONFIG.persistent_cache_ttl_sec)
