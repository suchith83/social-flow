"""
Unit Tests for Redis Module
"""

import pytest
from config.databases.redis.redis_connection import get_redis
from config.databases.redis.redis_cache import RedisCache
from config.databases.redis.redis_lock import RedisLock
from config.databases.redis.redis_utils import is_healthy


def test_connection():
    client = get_redis()
    assert client.ping() is True


def test_cache_set_get_delete():
    cache = RedisCache("test")
    key = "hello"
    value = {"foo": "bar"}
    assert cache.set(key, value, ttl=5)
    result = cache.get(key)
    assert result == value
    assert cache.exists(key)
    assert cache.delete(key)


def test_lock():
    lock = RedisLock("unit-test")
    assert lock.acquire(blocking=False)
    assert lock.release()


def test_health():
    assert is_healthy()
