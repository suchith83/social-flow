# Database-level caching strategies
import asyncio
import time
from typing import Any, Optional


class QueryCache:
    """
    Query cache with TTL.
    """

    def __init__(self):
        self.cache = {}
        self.lock = asyncio.Lock()

    async def set(self, key: str, value: Any, ttl: int = 60):
        expiry = time.time() + ttl
        async with self.lock:
            self.cache[key] = (value, expiry)

    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            item = self.cache.get(key)
            if not item:
                return None
            value, expiry = item
            if expiry < time.time():
                del self.cache[key]
                return None
            return value

    async def invalidate(self, key: str):
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
