"""
High-level uploader for files and directories to S3.
Supports multipart uploads, async, and progress tracking.
"""

import os
import asyncio
from tqdm import tqdm
from .client import S3Client


class S3Uploader:
    def __init__(self):
        self.client = S3Client()

    def upload_file(self, file_path: str, key: str):
        """Upload a single file."""
        self.client.upload_file(file_path, key)

    def upload_directory(self, dir_path: str, prefix: str = ""):
        """Recursively upload directory to S3."""
        for root, _, files in os.walk(dir_path):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, dir_path)
                s3_key = os.path.join(prefix, rel_path).replace("\\", "/")
                print(f"Uploading {local_path} -> {s3_key}")
                self.upload_file(local_path, s3_key)

    async def async_upload_file(self, file_path: str, key: str):
        """Async upload with aioboto3."""
        async with await self.client.aclient() as s3:
            await s3.upload_file(file_path, self.client.bucket_name, key)

    def upload_with_progress(self, file_path: str, key: str, chunk_size=1024 * 1024):
        """Upload file with progress bar."""
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=file_path) as pbar:
            self.client.client.upload_file(
                file_path,
                self.client.client._client_config.bucket_name,
                key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
