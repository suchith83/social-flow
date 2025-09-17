"""
Storage backend abstraction for raw uploads.
Supports:
- direct presigned S3 upload (recommended)
- server-side upload (using boto3) for staged files
- local filesystem staging
"""

import os
import boto3
from botocore.client import Config as BotoConfig
from typing import Tuple
from .config import config
from .utils import ensure_dir, logger

ensure_dir(config.STAGING_DIR)


class StorageBackend:
    def __init__(self):
        if config.STORAGE_BACKEND == "s3":
            self.s3 = boto3.client(
                "s3",
                endpoint_url=config.S3_ENDPOINT,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                config=BotoConfig(signature_version="s3v4"),
            )
            # ensure bucket exists (best effort)
            try:
                self.s3.head_bucket(Bucket=config.S3_BUCKET)
            except Exception:
                try:
                    self.s3.create_bucket(Bucket=config.S3_BUCKET)
                except Exception as e:
                    logger.warning(f"Could not create/verify bucket: {e}")
        else:
            self.s3 = None

    def presign_upload(self, key: str, expires_in: int = None) -> dict:
        """
        Generate presigned PUT URL for a given key.
        """
        if not self.s3:
            raise RuntimeError("S3 backend not configured")
        expires = expires_in or config.PRESIGNED_URL_TTL
        url = self.s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": config.S3_BUCKET, "Key": key},
            ExpiresIn=expires,
            HttpMethod="PUT",
        )
        return {"url": url, "expires_in": expires}

    def upload_file(self, file_path: str, key: str) -> str:
        """
        Server side upload of staged file to S3.
        Returns external URL.
        """
        if self.s3:
            self.s3.upload_file(file_path, config.S3_BUCKET, key)
            # construct object URL (compatible with MinIO)
            return f"{config.S3_ENDPOINT.rstrip('/')}/{config.S3_BUCKET}/{key}"
        # fallback: move to staging dir and return local path
        dest = os.path.join(config.STAGING_DIR, key)
        ensure_dir(os.path.dirname(dest))
        os.replace(file_path, dest)
        return f"file://{dest}"
