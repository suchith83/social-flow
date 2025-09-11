# common/libraries/python/database/cache.py
"""
Cache layer: in-memory with Redis-ready interface.
"""

import time
from typing import Any, Optional
from .config import DatabaseConfig

class MemoryCache:
    def __init__(self):
        self._store = {}

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expiry = int(time.time()) + (ttl or DatabaseConfig.CACHE_TTL)
        self._store[key] = (value, expiry)

    def get(self, key: str):
        value = self._store.get(key)
        if not value:
            return None
        val, expiry = value
        if expiry < int(time.time()):
            self._store.pop(key, None)
            return None
        return val

    def delete(self, key: str):
        self._store.pop(key, None)

cache_backend = MemoryCache()
