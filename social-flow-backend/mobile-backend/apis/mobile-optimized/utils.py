# Helpers (caching, response shaping, compression)
"""
Helper utilities for Mobile Optimized APIs.
Includes caching, compression, and response shaping.
"""

import functools
import time
import hashlib
from typing import Callable, Any, Dict

_cache_store: Dict[str, tuple] = {}


def cache_result(ttl: int = 60):
    """
    Simple in-memory cache decorator with TTL.
    Suitable for lightweight mobile endpoints.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = hashlib.sha256(f"{func.__name__}:{args}:{kwargs}".encode()).hexdigest()
            now = time.time()
            if key in _cache_store:
                value, expiry = _cache_store[key]
                if expiry > now:
                    return value
            result = func(*args, **kwargs)
            _cache_store[key] = (result, now + ttl)
            return result
        return wrapper
    return decorator
