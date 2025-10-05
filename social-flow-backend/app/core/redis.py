"""
Redis configuration and connection management.

This module handles Redis connections, caching, and session management
for the Social Flow backend.
"""

import json
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level Redis connection instances
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


async def init_redis() -> None:
    """Initialize Redis connection pool."""
    global _redis_pool, _redis_client
    
    # Skip Redis initialization in test mode
    if settings.TESTING:
        logger.info("Skipping Redis initialization in test mode")
        return
    
    try:
        _redis_pool = ConnectionPool.from_url(
            str(settings.REDIS_URL),
            max_connections=20,
            retry_on_timeout=True,
            decode_responses=True,
        )
        
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        
        # Test connection
        await _redis_client.ping()
        
        logger.info("Redis initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        logger.warning("Application will continue but caching and session features may be limited")
        # Set clients to None to indicate Redis is unavailable
        _redis_pool = None
        _redis_client = None


async def get_redis() -> Optional[redis.Redis]:
    """
    Get Redis client instance.
    
    Returns:
        redis.Redis: Redis client or None in test mode
    """
    # Return None in test mode
    if settings.TESTING:
        return None
    
    if _redis_client is None:
        await init_redis()
    
    return _redis_client


async def close_redis() -> None:
    """Close Redis connections."""
    global _redis_pool, _redis_client
    
    try:
        if _redis_client:
            await _redis_client.close()
            _redis_client = None
        if _redis_pool:
            await _redis_pool.disconnect()
            _redis_pool = None
        
        logger.info("Redis connections closed")
    except Exception as e:
        logger.error(f"Error closing Redis connections: {e}")
        raise


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol defining the cache contract."""
    
    async def get(self, key: str) -> Optional[Any]:
        ...
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        ...
    
    async def delete(self, key: str) -> bool:
        ...
    
    async def exists(self, key: str) -> bool:
        ...
    
    async def increment(self, key: str, amount: int = 1) -> int:
        ...
    
    async def expire(self, key: str, seconds: int) -> bool:
        ...


class RedisCache:
    """Redis cache utility class backed by Redis."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        try:
            serialized_value = json.dumps(value, default=str)
            return await self.redis.set(key, serialized_value, ex=expire)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            return await self.redis.delete(key) > 0
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache."""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        try:
            return await self.redis.expire(key, seconds)
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False


# In-memory cache fallback for test environments or degraded Redis
class InMemoryCache:
    """Simplified in-memory cache used when Redis is unavailable."""
    
    def __init__(self):
        self._store: Dict[str, Any] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        value = self._store.get(key)
        if value is None:
            return None
        return json.loads(value)
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        # Expiration is ignored in the simplified fallback implementation
        self._store[key] = json.dumps(value, default=str)
        return True
    
    async def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None
    
    async def exists(self, key: str) -> bool:
        return key in self._store
    
    async def increment(self, key: str, amount: int = 1) -> int:
        current_value = int(json.loads(self._store.get(key, json.dumps(0))))
        new_value = current_value + amount
        self._store[key] = json.dumps(new_value)
        return new_value
    
    async def expire(self, key: str, seconds: int) -> bool:
        # No-op for in-memory fallback
        return key in self._store


# Global cache instance
cache: Optional[CacheBackend] = None


async def get_cache() -> CacheBackend:
    """Get cache instance."""
    global cache
    
    if cache is None:
        client = await get_redis()
        if client is None:
            cache = InMemoryCache()
        else:
            cache = RedisCache(client)
    
    return cache
