"""
Downloader module for Google Cloud Storage.
"""

import os
import asyncio
from tqdm import tqdm
from .client import GCSClient


class GCSDownloader:
    def __init__(self):
        self.client = GCSClient()

    def download_file(self, blob_name: str, file_path: str):
        """Download a single blob to file."""
        self.client.download_blob(blob_name, file_path)

    async def async_download_file(self, blob_name: str, file_path: str):
        """Async download using gcsfs."""
        async with await self.client.afs() as fs:
            async with fs.open(f"{self.client.bucket.name}/{blob_name}", "rb") as f:
                data = await f.read()
                with open(file_path, "wb") as out:
                    out.write(data)

    def download_with_progress(self, blob_name: str, file_path: str, chunk_size=4 * 1024 * 1024):
        """Download with progress bar."""
        blob = self.client.bucket.blob(blob_name)
        total_size = blob.size
        with open(file_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=blob_name) as pbar:
            stream = blob.download_as_bytes()
            for i in range(0, len(stream), chunk_size):
                f.write(stream[i : i + chunk_size])
                pbar.update(min(chunk_size, len(stream) - i))
