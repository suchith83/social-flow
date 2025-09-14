# cdn_config.py
# Created by Create-Configurations.ps1
"""
cdn_config.py
-------------
Defines CDN cache configurations like edge TTL, cache invalidation strategies,
geo-distribution, and signed URL rules.
"""

import os
from typing import Dict


class CDNConfig:
    """Configuration handler for CDN caching."""

    DEFAULT_EDGE_TTL = int(os.getenv("CDN_EDGE_TTL", "300"))  # 5 min
    MAX_EDGE_TTL = int(os.getenv("CDN_MAX_EDGE_TTL", "86400"))  # 24h
    INVALIDATION_METHOD = os.getenv("CDN_INVALIDATION", "path")  # path | wildcard | regex
    SIGNED_URLS = os.getenv("CDN_SIGNED_URLS", "false").lower() == "true"
    GEO_DISTRIBUTION = os.getenv("CDN_GEO_DIST", "multi-region")

    @classmethod
    def summary(cls) -> Dict:
        return {
            "default_edge_ttl": cls.DEFAULT_EDGE_TTL,
            "max_edge_ttl": cls.MAX_EDGE_TTL,
            "invalidation_method": cls.INVALIDATION_METHOD,
            "signed_urls": cls.SIGNED_URLS,
            "geo_distribution": cls.GEO_DISTRIBUTION,
        }


if __name__ == "__main__":
    print("🔧 CDN Config:", CDNConfig.summary())
