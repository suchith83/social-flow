"""
Bucket management utilities for GCS.
"""

from .client import GCSClient


class GCSBucketManager:
    def __init__(self):
        self.client = GCSClient()

    def create_bucket(self, name: str, location: str = "US"):
        return self.client.client.create_bucket(name, location=location)

    def delete_bucket(self, name: str):
        bucket = self.client.client.bucket(name)
        bucket.delete()

    def list_buckets(self):
        return [b.name for b in self.client.client.list_buckets()]

    def list_blobs(self, prefix: str = ""):
        return [b.name for b in self.client.bucket.list_blobs(prefix=prefix)]
