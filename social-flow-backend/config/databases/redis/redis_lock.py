"""Distributed locking using Redlock algorithm."""
"""
Redis Distributed Lock (Redlock Implementation)
"""

import time
import uuid
from typing import Optional
from .redis_connection import get_redis


class RedisLock:
    def __init__(self, name: str, ttl: int = 10000):
        self.client = get_redis()
        self.name = f"lock:{name}"
        self.ttl = ttl
        self.token: Optional[str] = None

    def acquire(self, blocking: bool = True, retry_delay: float = 0.1) -> bool:
        """Acquire distributed lock."""
        self.token = str(uuid.uuid4())
        while True:
            if self.client.set(self.name, self.token, nx=True, px=self.ttl):
                return True
            if not blocking:
                return False
            time.sleep(retry_delay)

    def release(self) -> bool:
        """Release lock safely (only if owned)."""
        if not self.token:
            return False
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        result = self.client.eval(lua_script, 1, self.name, self.token)
        return result == 1
