# Implements cache tuning strategies for better performance
# performance/cdn/optimization/cache_tuning.py
"""
Cache tuning and optimization strategies.
"""

import time
from typing import Dict, Optional
from .utils import logger, timeit

class CacheManager:
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl

    @timeit
    def put(self, key: str, value: bytes, ttl: Optional[int] = None):
        """Store value in cache with TTL."""
        expire = time.time() + (ttl or self.default_ttl)
        self.cache[key] = {"value": value, "expire": expire}
        logger.debug(f"Cached key={key} ttl={ttl or self.default_ttl}")

    @timeit
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve value if valid; otherwise evict."""
        entry = self.cache.get(key)
        if not entry:
            return None
        if time.time() > entry["expire"]:
            logger.debug(f"Cache expired for {key}")
            del self.cache[key]
            return None
        return entry["value"]

    def invalidate(self, key: str):
        """Force remove from cache."""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated {key}")

    def auto_prefetch(self, hot_keys: list[str], fetch_fn):
        """
        Prefetch hot content into cache.
        `fetch_fn` should accept a key and return bytes.
        """
        for k in hot_keys:
            if k not in self.cache:
                try:
                    data = fetch_fn(k)
                    self.put(k, data)
                    logger.info(f"Prefetched {k}")
                except Exception as e:
                    logger.error(f"Prefetch failed for {k}: {e}")
