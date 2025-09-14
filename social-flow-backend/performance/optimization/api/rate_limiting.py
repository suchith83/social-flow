# Implements rate limiting logic for API requests
import time
import asyncio
import threading
from collections import deque
from typing import Optional


class TokenBucketRateLimiter:
    """
    Advanced Token Bucket Rate Limiter.

    - Allows bursts up to bucket size.
    - Replenishes tokens at a fixed rate.
    - Thread-safe and asyncio-compatible.
    """

    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def _add_tokens(self):
        now = time.monotonic()
        delta = now - self.timestamp
        added_tokens = delta * self.rate
        if added_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + added_tokens)
            self.timestamp = now

    def allow(self) -> bool:
        """Check if request is allowed."""
        with self.lock:
            self._add_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def allow_async(self) -> bool:
        """Async version of allow."""
        return self.allow()


class SlidingWindowRateLimiter:
    """
    Sliding Window Log Rate Limiter.

    - Maintains deque of timestamps.
    - Efficiently removes expired requests.
    """

    def __init__(self, limit: int, window_size: float):
        self.limit = limit
        self.window_size = window_size
        self.timestamps = deque()
        self.lock = asyncio.Lock()

    async def allow(self) -> bool:
        now = time.monotonic()
        async with self.lock:
            while self.timestamps and self.timestamps[0] <= now - self.window_size:
                self.timestamps.popleft()
            if len(self.timestamps) < self.limit:
                self.timestamps.append(now)
                return True
            return False
