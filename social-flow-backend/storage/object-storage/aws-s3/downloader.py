"""
Downloader module for S3.
Supports async download and progress tracking.
"""

import os
import asyncio
from tqdm import tqdm
from .client import S3Client


class S3Downloader:
    def __init__(self):
        self.client = S3Client()

    def download_file(self, key: str, file_path: str):
        """Download single file."""
        self.client.download_file(key, file_path)

    async def async_download_file(self, key: str, file_path: str):
        """Async download."""
        async with await self.client.aclient() as s3:
            await s3.download_file(self.client.bucket_name, key, file_path)

    def download_with_progress(self, key: str, file_path: str):
        """Download file with progress bar."""
        meta = self.client.client.head_object(Bucket=self.client.bucket_name, Key=key)
        total_size = meta["ContentLength"]

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=key) as pbar:
            self.client.client.download_file(
                self.client.bucket_name,
                key,
                file_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
