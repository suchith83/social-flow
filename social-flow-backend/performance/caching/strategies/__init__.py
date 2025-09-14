"""
Caching Strategies Module

This package implements multiple advanced caching strategies:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- FIFO (First In First Out)
- TTL (Time To Live)
- ARC (Adaptive Replacement Cache)

Also provides a CacheManager for flexible usage.
"""

__all__ = [
    "lru_cache",
    "lfu_cache",
    "fifo_cache",
    "ttl_cache",
    "arc_cache",
    "cache_manager",
]
