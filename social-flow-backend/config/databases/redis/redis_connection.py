"""Secure Redis connection pool manager."""
"""
Redis Connection Manager
- Provides connection pooling and retry logic.
- Supports SSL/TLS for secure environments.
"""

import redis
from redis.exceptions import ConnectionError, TimeoutError
import time
from typing import Optional
from .redis_config import get_config


class RedisConnectionManager:
    """Manages a Redis connection pool with retries."""

    def __init__(self):
        cfg = get_config()
        self.pool = redis.ConnectionPool(**cfg.as_dict())
        self.client = redis.Redis(connection_pool=self.pool)

    def get_client(self) -> redis.Redis:
        """Returns the Redis client instance."""
        return self.client

    def ping(self, retries: int = 3, delay: int = 2) -> bool:
        """Check if Redis is alive with retry support."""
        for attempt in range(1, retries + 1):
            try:
                return self.client.ping()
            except (ConnectionError, TimeoutError):
                if attempt == retries:
                    raise
                time.sleep(delay)
        return False


# Singleton accessor
_connection_manager: Optional[RedisConnectionManager] = None


def get_redis() -> redis.Redis:
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = RedisConnectionManager()
    return _connection_manager.get_client()
