"""Helper utilities for Redis operations."""
"""
Redis Utility Functions
- Key namespacing
- Health checks
- Metrics gathering
"""

from typing import Dict, Any
from .redis_connection import get_redis


def namespaced_key(namespace: str, key: str) -> str:
    return f"{namespace}:{key}"


def get_info() -> Dict[str, Any]:
    """Get Redis server information."""
    client = get_redis()
    return client.info()


def is_healthy() -> bool:
    """Check if Redis is alive."""
    try:
        return get_redis().ping()
    except Exception:
        return False
