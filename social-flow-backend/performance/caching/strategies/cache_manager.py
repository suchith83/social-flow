from .lru_cache import LRUCache
from .lfu_cache import LFUCache
from .fifo_cache import FIFOCache
from .ttl_cache import TTLCache
from .arc_cache import ARCCache

class CacheManager:
    """
    Unified interface to switch between caching strategies.
    """

    STRATEGIES = {
        "lru": LRUCache,
        "lfu": LFUCache,
        "fifo": FIFOCache,
        "ttl": TTLCache,
        "arc": ARCCache,
    }

    def __init__(self, strategy: str = "lru", **kwargs):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unsupported cache strategy: {strategy}")
        self.cache = self.STRATEGIES[strategy](**kwargs)

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        self.cache.put(key, value)

    def __repr__(self):
        return f"CacheManager({self.cache})"
