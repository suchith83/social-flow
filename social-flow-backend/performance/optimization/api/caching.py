# Provides caching strategies for API responses
import asyncio
import time
from typing import Any, Dict, Optional


class InMemoryCache:
    """
    In-memory cache with TTL support.
    Async-safe with asyncio.Lock.
    """

    def __init__(self):
        self.store: Dict[str, tuple[Any, float]] = {}
        self.lock = asyncio.Lock()

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expiry = time.time() + ttl if ttl else float("inf")
        async with self.lock:
            self.store[key] = (value, expiry)

    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            item = self.store.get(key)
            if not item:
                return None
            value, expiry = item
            if expiry < time.time():
                del self.store[key]
                return None
            return value

    async def delete(self, key: str):
        async with self.lock:
            if key in self.store:
                del self.store[key]


class DistributedCache:
    """
    Simulated Distributed Cache Layer.

    - In practice, integrates with Redis/Memcached.
    - Async interface for consistency.
    """

    def __init__(self):
        self.local_cache = InMemoryCache()

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        await self.local_cache.set(key, value, ttl)

    async def get(self, key: str) -> Optional[Any]:
        return await self.local_cache.get(key)

    async def delete(self, key: str):
        await self.local_cache.delete(key)
