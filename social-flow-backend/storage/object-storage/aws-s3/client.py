"""
S3 Client wrapper with retry logic and both sync/async support.
"""

import boto3
import aioboto3
import botocore
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import s3_config


class S3Client:
    """Wrapper around boto3 and aioboto3 clients with retry logic."""

    def __init__(self):
        self._client = boto3.client(
            "s3",
            aws_access_key_id=s3_config.aws_access_key_id,
            aws_secret_access_key=s3_config.aws_secret_access_key,
            region_name=s3_config.aws_region,
        )

    @property
    def client(self):
        """Return boto3 client."""
        return self._client

    def resource(self):
        """Return boto3 resource."""
        return boto3.resource(
            "s3",
            aws_access_key_id=s3_config.aws_access_key_id,
            aws_secret_access_key=s3_config.aws_secret_access_key,
            region_name=s3_config.aws_region,
        )

    async def aclient(self):
        """Return aioboto3 async client."""
        session = aioboto3.Session()
        return session.client(
            "s3",
            aws_access_key_id=s3_config.aws_access_key_id,
            aws_secret_access_key=s3_config.aws_secret_access_key,
            region_name=s3_config.aws_region,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def upload_file(self, file_path: str, key: str):
        """Upload file with retries."""
        try:
            self._client.upload_file(file_path, s3_config.bucket_name, key)
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Upload failed: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def download_file(self, key: str, file_path: str):
        """Download file with retries."""
        try:
            self._client.download_file(s3_config.bucket_name, key, file_path)
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Download failed: {e}")
