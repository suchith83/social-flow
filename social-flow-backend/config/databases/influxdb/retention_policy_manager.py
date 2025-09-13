"""Create and manage retention policies."""
"""
retention_policy_manager.py
---------------------------
Manages retention policies for InfluxDB buckets.
"""

from .connection import InfluxDBConnection
import logging

logger = logging.getLogger("RetentionPolicyManager")
logger.setLevel(logging.INFO)


class RetentionPolicyManager:
    def __init__(self):
        self.client = InfluxDBConnection().get_client()
        self.bucket_api = self.client.buckets_api()
        self.org = "socialflow-org"

    def create_bucket(self, name: str, retention_days: int):
        """Create a bucket with retention policy."""
        rp_seconds = retention_days * 24 * 60 * 60
        bucket = self.bucket_api.create_bucket(bucket_name=name, org=self.org, retention_rules=[{"everySeconds": rp_seconds}])
        logger.info(f"✅ Bucket {name} created with {retention_days} days retention.")
        return bucket

    def update_retention(self, bucket_id: str, retention_days: int):
        """Update bucket retention policy."""
        rp_seconds = retention_days * 24 * 60 * 60
        bucket = self.bucket_api.find_bucket_by_id(bucket_id)
        bucket.retention_rules[0].every_seconds = rp_seconds
        self.bucket_api.update_bucket(bucket)
        logger.info(f"♻️ Retention updated to {retention_days} days for {bucket.name}.")
