"""
# High-level orchestration of CloudFront distributions
"""
# config/cdn/aws-cloudfront/cloudfront_manager.py
"""
CloudFront Manager
------------------
High-level orchestration for managing AWS CloudFront distributions.
"""

import boto3
from typing import Dict, Any


class CloudFrontManager:
    def __init__(self, session=None):
        self.session = session or boto3.Session()
        self.client = self.session.client("cloudfront")

    def create_distribution(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a CloudFront distribution with provided config."""
        response = self.client.create_distribution(DistributionConfig=config)
        return response["Distribution"]

    def update_distribution(self, dist_id: str, etag: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update distribution (requires etag for concurrency safety)."""
        response = self.client.update_distribution(
            Id=dist_id,
            IfMatch=etag,
            DistributionConfig=config,
        )
        return response["Distribution"]

    def delete_distribution(self, dist_id: str, etag: str) -> bool:
        """Delete a distribution (must be disabled first)."""
        self.client.delete_distribution(Id=dist_id, IfMatch=etag)
        return True

    def get_distribution(self, dist_id: str) -> Dict[str, Any]:
        """Fetch distribution details."""
        return self.client.get_distribution(Id=dist_id)["Distribution"]

    def list_distributions(self) -> Dict[str, Any]:
        """List all CloudFront distributions."""
        return self.client.list_distributions()["DistributionList"]
