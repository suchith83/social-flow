# redis_invalidator.py
# Created by Create-Invalidation.ps1
"""
redis_invalidator.py
--------------------
A Redis-based invalidator that supports:
- single-key deletes
- pattern deletes (careful with KEYS; uses SCAN)
- pipelined/batched deletes
- mark_stale via short TTL
- set_ttl for scheduled expiration
- safe retries and exponential backoff
"""

from __future__ import annotations
import time
import logging
from typing import Iterable, List, Optional
import redis
from .exceptions import RemoteInvalidateError

logger = logging.getLogger(__name__)


def _simple_retry(action_fn, retries: int = 3, base_delay: float = 0.1):
    """Simple exponential-backoff retry wrapper for IO ops."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return action_fn()
        except Exception as e:
            last_exc = e
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning("Redis operation failed (attempt %d/%d): %s, retrying in %.2fs", attempt, retries, e, wait)
            time.sleep(wait)
    raise last_exc


class RedisInvalidator:
    """
    Accepts a redis.Redis instance (or connection pool) and exposes invalidation helpers.

    Note: For cluster mode, pass a redis.RedisCluster client configured accordingly.
    """

    def __init__(self, client: redis.Redis, scan_batch: int = 1000):
        self.client = client
        self.scan_batch = scan_batch

    def invalidate(self, keys: Iterable[str]) -> None:
        """
        Delete keys in a pipeline. Accepts explicit keys; patterns should be passed to invalidate_pattern.
        """
        keys_list = list(keys)
        if not keys_list:
            return

        def _do():
            pipe = self.client.pipeline(transaction=False)
            for k in keys_list:
                pipe.delete(k)
            results = pipe.execute()
            logger.debug("RedisInvalidator.delete results: %s", results)
            return results

        try:
            _simple_retry(_do)
            logger.info("Invalidated %d keys in Redis", len(keys_list))
        except Exception as e:
            logger.exception("Failed to invalidate keys in Redis")
            raise RemoteInvalidateError(str(e))

    def invalidate_pattern(self, pattern: str) -> None:
        """
        Delete keys matching a pattern using SCAN to avoid blocking Redis.
        """
        logger.info("Starting pattern invalidation for pattern=%s", pattern)

        def _do_scan_delete():
            cursor = "0"
            total = 0
            while cursor != 0:
                cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=self.scan_batch)
                if keys:
                    pipe = self.client.pipeline(transaction=False)
                    for k in keys:
                        pipe.delete(k)
                    pipe.execute()
                    total += len(keys)
            return total

        try:
            deleted = _simple_retry(_do_scan_delete)
            logger.info("Pattern invalidation removed %d keys", deleted)
        except Exception as e:
            logger.exception("Pattern invalidation failed")
            raise RemoteInvalidateError(str(e))

    def mark_stale(self, keys: Iterable[str], ttl: int = 60) -> None:
        """Set a short TTL to mark keys stale; many cache clients interpret TTL as seconds."""
        keys_list = list(keys)
        if not keys_list:
            return

        def _do():
            pipe = self.client.pipeline(transaction=False)
            for k in keys_list:
                pipe.expire(k, ttl)
            return pipe.execute()

        try:
            _simple_retry(_do)
            logger.debug("Marked %d keys stale (ttl=%ds)", len(keys_list), ttl)
        except Exception as e:
            logger.exception("Failed to mark stale")
            raise RemoteInvalidateError(str(e))

    def set_ttl(self, keys: Iterable[str], ttl: int) -> None:
        """Set TTL to a specific value (0 for immediate expiration where supported)."""
        # Some clients will treat ttl=0 as immediate delete; we map 0 -> delete for compatibility.
        if ttl == 0:
            self.invalidate(keys)
            return
        self.mark_stale(keys, ttl=ttl)
