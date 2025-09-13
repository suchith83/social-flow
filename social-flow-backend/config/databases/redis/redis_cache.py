"""High-level caching utilities using Redis."""
"""
Redis Cache Abstraction
- Provides high-level cache operations (JSON serialization, TTL, compression).
"""

import json
import zlib
from typing import Any, Optional
from .redis_connection import get_redis


class RedisCache:
    def __init__(self, namespace: str = "app:cache"):
        self.client = get_redis()
        self.namespace = namespace

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def set(
        self, key: str, value: Any, ttl: int = 3600, compress: bool = False
    ) -> bool:
        """Set a value in cache with TTL."""
        data = json.dumps(value).encode("utf-8")
        if compress:
            data = zlib.compress(data)
        return bool(self.client.setex(self._key(key), ttl, data))

    def get(self, key: str, decompress: bool = False) -> Optional[Any]:
        """Retrieve value from cache."""
        data = self.client.get(self._key(key))
        if not data:
            return None
        if decompress:
            data = zlib.decompress(data)
        return json.loads(data.decode("utf-8"))

    def delete(self, key: str) -> bool:
        """Delete a cached value."""
        return bool(self.client.delete(self._key(key)))

    def exists(self, key: str) -> bool:
        return bool(self.client.exists(self._key(key)))
