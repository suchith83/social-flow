# Configuration for pagination, cache, compression
"""
Configuration for Mobile Optimized APIs.
Handles caching, response limits, and compression settings.
"""

import os
from functools import lru_cache


class MobileOptimizedConfig:
    MAX_PAGE_SIZE: int = int(os.getenv("MAX_PAGE_SIZE", 50))
    DEFAULT_PAGE_SIZE: int = int(os.getenv("DEFAULT_PAGE_SIZE", 20))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 60))  # in seconds
    ENABLE_COMPRESSION: bool = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"


@lru_cache()
def get_config() -> MobileOptimizedConfig:
    return MobileOptimizedConfig()
