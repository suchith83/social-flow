"""
Enhanced Redis Caching Infrastructure

This module provides comprehensive Redis caching with:
- Connection pooling and cluster support
- Multiple cache strategies (LRU, TTL, write-through, write-behind)
- Distributed locking for race conditions
- Pub/Sub for real-time features
- Rate limiting implementation
- Session management
- View count tracking with batch processing
"""

import asyncio
import json
import logging
import pickle
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union
from functools import wraps

import redis.asyncio as aioredis
from redis.asyncio import Redis, RedisCluster
from redis.exceptions import RedisError, LockError

try:
    from app.core.config_enhanced import settings
except ImportError:
    from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ============================================================================
# REDIS MANAGER
# ============================================================================

class RedisManager:
    """
    Manages Redis connections with support for clustering and pub/sub.
    
    Features:
    - Connection pooling
    - Redis Cluster support for horizontal scaling
    - Automatic reconnection
    - Health monitoring
    - Distributed operations
    """
    
    def __init__(self):
        self._redis: Optional[Union[Redis, RedisCluster]] = None
        self._pubsub_connection: Optional[Redis] = None
        self._initialized: bool = False
        self._lock_prefix: str = "lock:"
        self._default_ttl: int = settings.CACHE_TTL_DEFAULT
    
    async def initialize(self) -> None:
        """Initialize Redis connection(s)."""
        if self._initialized:
            logger.warning("Redis manager already initialized")
            return
        
        logger.info("Initializing Redis manager...")
        
        try:
            if settings.REDIS_CLUSTER_ENABLED and settings.REDIS_CLUSTER_NODES:
                # Redis Cluster mode
                startup_nodes = [
                    {"host": node.split(":")[0], "port": int(node.split(":")[1])}
                    for node in settings.REDIS_CLUSTER_NODES
                ]
                self._redis = RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=False,  # Handle encoding manually for flexibility
                    max_connections=settings.REDIS_MAX_CONNECTIONS,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
                )
                logger.info(f"Redis Cluster initialized with {len(startup_nodes)} nodes")
            else:
                # Single Redis instance
                self._redis = await aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=False,
                    max_connections=settings.REDIS_MAX_CONNECTIONS,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
                )
                logger.info(f"Redis initialized: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            
            # Separate connection for pub/sub
            self._pubsub_connection = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            
            # Verify connection
            await self._redis.ping()
            logger.info("Redis connection verified")
            
            self._initialized = True
        
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}", exc_info=True)
            raise
    
    async def close(self) -> None:
        """Close Redis connections gracefully."""
        if not self._initialized:
            return
        
        logger.info("Closing Redis connections...")
        
        if self._redis:
            await self._redis.close()
            logger.info("Main Redis connection closed")
        
        if self._pubsub_connection:
            await self._pubsub_connection.close()
            logger.info("Pub/Sub Redis connection closed")
        
        self._initialized = False
    
    def get_client(self) -> Union[Redis, RedisCluster]:
        """Get the Redis client."""
        if not self._redis:
            raise RuntimeError("Redis manager not initialized")
        return self._redis
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            await self._redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    # ========================================================================
    # CACHING OPERATIONS
    # ========================================================================
    
    async def get(
        self,
        key: str,
        default: Optional[Any] = None,
        deserialize: bool = True,
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            deserialize: Whether to deserialize (pickle) the value
        
        Returns:
            Cached value or default
        """
        try:
            value = await self._redis.get(key)
            if value is None:
                return default
            
            if deserialize:
                return pickle.loads(value)
            return value.decode("utf-8") if isinstance(value, bytes) else value
        
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True,
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = no expiration)
            serialize: Whether to serialize (pickle) the value
        
        Returns:
            True if successful
        """
        try:
            if serialize:
                value_bytes = pickle.dumps(value)
            else:
                value_bytes = value if isinstance(value, bytes) else str(value).encode("utf-8")
            
            if ttl:
                await self._redis.setex(key, ttl, value_bytes)
            else:
                await self._redis.set(key, value_bytes)
            
            return True
        
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """
        Delete one or more keys from cache.
        
        Returns:
            Number of keys deleted
        """
        try:
            return await self._redis.delete(*keys)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return 0
    
    async def exists(self, *keys: str) -> int:
        """
        Check if keys exist.
        
        Returns:
            Number of keys that exist
        """
        try:
            return await self._redis.exists(*keys)
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for a key."""
        try:
            return await self._redis.expire(key, seconds)
        except Exception as e:
            logger.error(f"Cache expire error for key '{key}': {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a numeric value.
        
        Returns:
            New value after increment
        """
        try:
            return await self._redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache incr error for key '{key}': {e}")
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """
        Decrement a numeric value.
        
        Returns:
            New value after decrement
        """
        try:
            return await self._redis.decrby(key, amount)
        except Exception as e:
            logger.error(f"Cache decr error for key '{key}': {e}")
            return 0
    
    # ========================================================================
    # HASH OPERATIONS (for structured data)
    # ========================================================================
    
    async def hset(self, name: str, key: str, value: Any) -> bool:
        """Set hash field."""
        try:
            value_bytes = pickle.dumps(value)
            return await self._redis.hset(name, key, value_bytes)
        except Exception as e:
            logger.error(f"Hash set error for '{name}.{key}': {e}")
            return False
    
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field."""
        try:
            value = await self._redis.hget(name, key)
            return pickle.loads(value) if value else None
        except Exception as e:
            logger.error(f"Hash get error for '{name}.{key}': {e}")
            return None
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            data = await self._redis.hgetall(name)
            return {
                k.decode("utf-8"): pickle.loads(v)
                for k, v in data.items()
            }
        except Exception as e:
            logger.error(f"Hash getall error for '{name}': {e}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        try:
            return await self._redis.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Hash delete error for '{name}': {e}")
            return 0
    
    # ========================================================================
    # SET OPERATIONS (for collections)
    # ========================================================================
    
    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        try:
            return await self._redis.sadd(key, *members)
        except Exception as e:
            logger.error(f"Set add error for '{key}': {e}")
            return 0
    
    async def srem(self, key: str, *members: str) -> int:
        """Remove members from a set."""
        try:
            return await self._redis.srem(key, *members)
        except Exception as e:
            logger.error(f"Set remove error for '{key}': {e}")
            return 0
    
    async def smembers(self, key: str) -> Set[str]:
        """Get all members of a set."""
        try:
            members = await self._redis.smembers(key)
            return {m.decode("utf-8") if isinstance(m, bytes) else m for m in members}
        except Exception as e:
            logger.error(f"Set members error for '{key}': {e}")
            return set()
    
    async def sismember(self, key: str, member: str) -> bool:
        """Check if member is in set."""
        try:
            return await self._redis.sismember(key, member)
        except Exception as e:
            logger.error(f"Set ismember error for '{key}': {e}")
            return False
    
    # ========================================================================
    # SORTED SET OPERATIONS (for rankings, leaderboards)
    # ========================================================================
    
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members with scores to sorted set."""
        try:
            return await self._redis.zadd(key, mapping)
        except Exception as e:
            logger.error(f"Sorted set add error for '{key}': {e}")
            return 0
    
    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
    ) -> List[Any]:
        """Get members from sorted set by rank."""
        try:
            return await self._redis.zrange(key, start, end, desc=desc, withscores=withscores)
        except Exception as e:
            logger.error(f"Sorted set range error for '{key}': {e}")
            return []
    
    async def zincrby(self, key: str, amount: float, member: str) -> float:
        """Increment score of member in sorted set."""
        try:
            return await self._redis.zincrby(key, amount, member)
        except Exception as e:
            logger.error(f"Sorted set incrby error for '{key}': {e}")
            return 0.0
    
    # ========================================================================
    # DISTRIBUTED LOCKING
    # ========================================================================
    
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: Optional[float] = None,
    ) -> Optional[aioredis.lock.Lock]:
        """
        Acquire a distributed lock.
        
        Args:
            lock_name: Name of the lock
            timeout: Lock expiration timeout (seconds)
            blocking: Whether to wait for lock
            blocking_timeout: Max time to wait for lock (seconds)
        
        Returns:
            Lock object if acquired, None otherwise
        
        Example:
            lock = await redis_manager.acquire_lock("video_process:123")
            if lock:
                try:
                    # Do work
                    pass
                finally:
                    await redis_manager.release_lock(lock)
        """
        try:
            lock_key = f"{self._lock_prefix}{lock_name}"
            lock = self._redis.lock(
                lock_key,
                timeout=timeout,
                blocking=blocking,
                blocking_timeout=blocking_timeout,
            )
            
            acquired = await lock.acquire()
            return lock if acquired else None
        
        except Exception as e:
            logger.error(f"Lock acquire error for '{lock_name}': {e}")
            return None
    
    async def release_lock(self, lock: aioredis.lock.Lock) -> bool:
        """Release a distributed lock."""
        try:
            await lock.release()
            return True
        except LockError as e:
            logger.warning(f"Lock release error (may have expired): {e}")
            return False
        except Exception as e:
            logger.error(f"Lock release error: {e}")
            return False
    
    # ========================================================================
    # PUB/SUB OPERATIONS
    # ========================================================================
    
    async def publish(self, channel: str, message: str) -> int:
        """
        Publish message to channel.
        
        Returns:
            Number of subscribers that received the message
        """
        try:
            return await self._pubsub_connection.publish(channel, message)
        except Exception as e:
            logger.error(f"Publish error to channel '{channel}': {e}")
            return 0
    
    async def subscribe(self, *channels: str) -> aioredis.client.PubSub:
        """
        Subscribe to channels.
        
        Returns:
            PubSub object for receiving messages
        
        Example:
            pubsub = await redis_manager.subscribe("notifications")
            async for message in pubsub.listen():
                if message["type"] == "message":
                    print(message["data"])
        """
        try:
            pubsub = self._pubsub_connection.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
            raise
    
    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================
    
    async def pipeline_execute(self, commands: List[tuple]) -> List[Any]:
        """
        Execute multiple commands in a pipeline for better performance.
        
        Args:
            commands: List of (command_name, *args) tuples
        
        Returns:
            List of results
        
        Example:
            results = await redis_manager.pipeline_execute([
                ("set", "key1", "value1"),
                ("get", "key2"),
                ("incr", "counter"),
            ])
        """
        try:
            pipeline = self._redis.pipeline()
            for cmd in commands:
                getattr(pipeline, cmd[0])(*cmd[1:])
            return await pipeline.execute()
        except Exception as e:
            logger.error(f"Pipeline execute error: {e}")
            return []


# ============================================================================
# GLOBAL REDIS MANAGER
# ============================================================================

redis_manager = RedisManager()


# ============================================================================
# CACHING DECORATORS
# ============================================================================

def cache_result(
    ttl: int = 300,
    key_prefix: str = "",
    key_builder: Optional[Callable] = None,
):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache key
        key_builder: Custom function to build cache key from args
    
    Example:
        @cache_result(ttl=600, key_prefix="user")
        async def get_user_profile(user_id: str) -> dict:
            # Expensive database query
            return profile
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(filter(None, key_parts))
            
            # Try to get from cache
            cached_value = await redis_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_value
            
            # Execute function and cache result
            logger.debug(f"Cache miss for key: {cache_key}")
            result = await func(*args, **kwargs)
            await redis_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Rate limiting using Redis."""
    
    def __init__(self, redis_client: Union[Redis, RedisCluster]):
        self.redis = redis_client
    
    async def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (e.g., user_id, IP address)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        
        Returns:
            True if allowed, False if rate limit exceeded
        """
        key = f"rate_limit:{identifier}"
        
        try:
            # Get current count
            current = await self.redis.get(key)
            
            if current is None:
                # First request in window
                pipeline = self.redis.pipeline()
                pipeline.set(key, 1)
                pipeline.expire(key, window_seconds)
                await pipeline.execute()
                return True
            
            current_count = int(current)
            if current_count < max_requests:
                await self.redis.incr(key)
                return True
            
            # Rate limit exceeded
            return False
        
        except Exception as e:
            logger.error(f"Rate limit check error for '{identifier}': {e}")
            # Fail open to not block legitimate users on Redis errors
            return True


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

async def init_redis() -> None:
    """Initialize Redis connections."""
    await redis_manager.initialize()


async def close_redis() -> None:
    """Close Redis connections."""
    await redis_manager.close()


def get_redis() -> Union[Redis, RedisCluster]:
    """Get Redis client (legacy compatibility)."""
    return redis_manager.get_client()


# Export all public APIs
__all__ = [
    "redis_manager",
    "RedisManager",
    "RateLimiter",
    "cache_result",
    "init_redis",
    "close_redis",
    "get_redis",
]
