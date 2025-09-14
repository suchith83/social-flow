# cache_settings.py
# Created by Create-Configurations.ps1
"""
cache_settings.py
-----------------
Centralized cache configuration definitions.
Defines default TTLs, eviction policies, sharding strategies, 
and dynamic overrides from environment variables or feature flags.
"""

import os
from enum import Enum


class EvictionPolicy(Enum):
    """Supported eviction policies across caches."""
    LRU = "least-recently-used"
    LFU = "least-frequently-used"
    FIFO = "first-in-first-out"
    ARC = "adaptive-replacement-cache"


class CacheSettings:
    """
    Global cache settings applied across services.
    These settings can be overridden per-cache (Redis, Memcached, CDN).
    """

    DEFAULT_TTL = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))  # 1 hour
    MAX_TTL = int(os.getenv("CACHE_MAX_TTL", "86400"))  # 24 hours
    EVICTION_POLICY = EvictionPolicy(
        os.getenv("CACHE_EVICTION_POLICY", "least-recently-used")
    )
    SHARD_COUNT = int(os.getenv("CACHE_SHARD_COUNT", "16"))
    COMPRESS_OBJECTS = os.getenv("CACHE_COMPRESS", "true").lower() == "true"

    @classmethod
    def summary(cls) -> dict:
        """Return the current cache configuration as a dictionary."""
        return {
            "default_ttl": cls.DEFAULT_TTL,
            "max_ttl": cls.MAX_TTL,
            "eviction_policy": cls.EVICTION_POLICY.value,
            "shard_count": cls.SHARD_COUNT,
            "compress_objects": cls.COMPRESS_OBJECTS,
        }


if __name__ == "__main__":
    # Debug: print cache configuration at runtime
    print("🔧 Cache Settings:", CacheSettings.summary())
