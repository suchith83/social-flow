"""
# Cache policies, invalidations
"""
# config/cdn/aws-cloudfront/cloudfront_cache.py
"""
CloudFront Cache
----------------
Manages cache behaviors, policies, and invalidations.
"""

import boto3
from typing import List


class CloudFrontCache:
    def __init__(self, session=None):
        self.session = session or boto3.Session()
        self.client = self.session.client("cloudfront")

    def invalidate_paths(self, dist_id: str, paths: List[str], caller_reference: str) -> str:
        """Invalidate cached paths in CloudFront."""
        response = self.client.create_invalidation(
            DistributionId=dist_id,
            InvalidationBatch={"Paths": {"Quantity": len(paths), "Items": paths}, "CallerReference": caller_reference},
        )
        return response["Invalidation"]["Id"]

    def set_min_ttl(self, config, ttl: int):
        """Set minimum TTL for cache behavior."""
        config["DefaultCacheBehavior"]["MinTTL"] = ttl
        return config

    def set_max_ttl(self, config, ttl: int):
        """Set maximum TTL for cache behavior."""
        config["DefaultCacheBehavior"]["MaxTTL"] = ttl
        return config
