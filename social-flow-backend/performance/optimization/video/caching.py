# Video content caching utilities
"""
caching.py

Segment-level caching implementation with TTL and size limits.

- SegmentCache supports LRU eviction and TTL-based invalidation.
- Designed for storing small binary segments (HLS .ts, fMP4 fragments).
- Thread-safe and asyncio-friendly.
"""

import asyncio
import time
from collections import OrderedDict
from typing import Optional


class SegmentCache:
    """
    LRU cache where keys are segment URIs and values are bytes.

    Parameters:
    - max_size_bytes: total bytes to keep cached (approximate)
    - default_ttl: default time-to-live for cached segments
    """

    def __init__(self, max_size_bytes: int = 100 * 1024 * 1024, default_ttl: int = 300):
        self.max_size_bytes = max_size_bytes
        self.default_ttl = default_ttl
        self.store = OrderedDict()  # key -> (bytes, expiry_ts, size)
        self.current_size = 0
        self.lock = asyncio.Lock()

    async def set(self, key: str, data: bytes, ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        size = len(data)
        async with self.lock:
            # If exists, remove old
            if key in self.store:
                _, old_exp, old_size = self.store.pop(key)
                self.current_size -= old_size
            # Evict until fits
            while self.current_size + size > self.max_size_bytes and self.store:
                _, (vdata, vexp, vsize) = self.store.popitem(last=False)
                self.current_size -= vsize
            self.store[key] = (data, expiry, size)
            self.current_size += size

    async def get(self, key: str) -> Optional[bytes]:
        async with self.lock:
            item = self.store.get(key)
            if not item:
                return None
            data, expiry, size = item
            if expiry < time.time():
                # stale; remove
                self.store.pop(key, None)
                self.current_size -= size
                return None
            # Move to end (recently used)
            self.store.move_to_end(key)
            return data

    async def invalidate(self, key: str):
        async with self.lock:
            item = self.store.pop(key, None)
            if item:
                _, _, size = item
                self.current_size -= size

    async def clear(self):
        async with self.lock:
            self.store.clear()
            self.current_size = 0
