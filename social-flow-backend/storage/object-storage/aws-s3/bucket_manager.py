"""
Bucket management utilities for S3.
"""

from .client import S3Client


class S3BucketManager:
    def __init__(self):
        self.client = S3Client()

    def create_bucket(self, bucket_name: str):
        return self.client.client.create_bucket(Bucket=bucket_name)

    def delete_bucket(self, bucket_name: str):
        return self.client.client.delete_bucket(Bucket=bucket_name)

    def list_buckets(self):
        return [b["Name"] for b in self.client.client.list_buckets()["Buckets"]]

    def list_objects(self, bucket_name: str, prefix: str = ""):
        resp = self.client.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
