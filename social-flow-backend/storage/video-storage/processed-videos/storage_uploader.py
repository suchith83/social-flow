"""
Upload processed files to object storage (S3/MinIO/GCS)
"""

import boto3
import os
from .config import config
from .utils import logger


class StorageUploader:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=config.STORAGE_ENDPOINT,
            aws_access_key_id=config.STORAGE_ACCESS_KEY,
            aws_secret_access_key=config.STORAGE_SECRET_KEY,
        )

    def upload(self, file_path: str, video_id: str) -> str:
        key = f"{video_id}/{os.path.basename(file_path)}"
        self.client.upload_file(file_path, config.STORAGE_BUCKET, key)
        url = f"{config.STORAGE_ENDPOINT}/{config.STORAGE_BUCKET}/{key}"
        logger.info(f"Uploaded {file_path} to {url}")
        return url
