# hybrid_cache_policy.py
# Created by Create-Configurations.ps1
"""
hybrid_cache_policy.py
----------------------
Defines hybrid caching strategies combining in-memory, distributed, and CDN caches.
"""

import os
from enum import Enum


class CacheLayer(Enum):
    MEMORY = "in-memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    CDN = "cdn"


class HybridCachePolicy:
    """Defines cache hierarchy and lookup order."""

    POLICY = os.getenv(
        "HYBRID_CACHE_POLICY", "memory>redis>cdn"
    ).split(">")

    @classmethod
    def get_layers(cls):
        """Return enabled layers in priority order."""
        return [CacheLayer(layer) for layer in cls.POLICY if layer in CacheLayer._value2member_map_]

    @classmethod
    def summary(cls) -> dict:
        return {"hybrid_policy": [layer.value for layer in cls.get_layers()]}


if __name__ == "__main__":
    print("🔧 Hybrid Cache Policy:", HybridCachePolicy.summary())
