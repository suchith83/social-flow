"""
Upload thumbnails to S3/MinIO or keep local.

Provides presigned URL generation and server-side upload helpers.
"""

import os
from botocore.client import Config as BotoConfig
import boto3
from typing import Dict
from .config import config
from .utils import ensure_dir, logger

ensure_dir(config.OUTPUT_DIR)


class ThumbnailStorage:
    def __init__(self):
        self.backend = config.STORAGE_BACKEND
        if self.backend == "s3":
            self.s3 = boto3.client(
                "s3",
                endpoint_url=config.S3_ENDPOINT,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                config=BotoConfig(signature_version="s3v4")
            )
            # ensure bucket exists best-effort
            try:
                self.s3.head_bucket(Bucket=config.S3_BUCKET)
            except Exception:
                try:
                    self.s3.create_bucket(Bucket=config.S3_BUCKET)
                except Exception as e:
                    logger.warning("Could not create S3 bucket: %s", e)
        else:
            self.s3 = None

    def upload_file(self, local_path: str, key: str) -> str:
        """Server-side upload (synchronous). Returns URL (or file:// path)."""
        if self.backend == "s3" and self.s3:
            self.s3.upload_file(local_path, config.S3_BUCKET, key)
            url = f"{config.S3_ENDPOINT.rstrip('/')}/{config.S3_BUCKET}/{key}"
            logger.info("Uploaded %s -> %s", local_path, url)
            return url
        else:
            # local fallback - move into organized output dir (already local)
            dest = os.path.join(config.OUTPUT_DIR, key)
            ensure_dir(os.path.dirname(dest))
            os.replace(local_path, dest)
            url = f"file://{dest}"
            logger.info("Moved %s -> %s", local_path, url)
            return url

    def presign_get(self, key: str, expires_in: int = None) -> Dict:
        if self.backend != "s3" or not self.s3:
            raise RuntimeError("Presign only supported for S3 backend")
        expires = expires_in or config.PRESIGNED_URL_TTL
        url = self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": config.S3_BUCKET, "Key": key},
            ExpiresIn=expires,
        )
        return {"url": url, "expires_in": expires}
