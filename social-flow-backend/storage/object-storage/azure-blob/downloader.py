"""
Downloader module for Azure Blob Storage.
Supports async download and progress.
"""

import os
import asyncio
from tqdm import tqdm
from .client import AzureBlobClient


class AzureBlobDownloader:
    def __init__(self):
        self.client = AzureBlobClient()

    def download_file(self, blob_name: str, file_path: str):
        """Download blob to file."""
        data = self.client.download_blob(blob_name)
        with open(file_path, "wb") as f:
            f.write(data)

    async def async_download_file(self, blob_name: str, file_path: str):
        """Async download."""
        async with await self.client.aclient() as client:
            container_client = client.get_container_client(self.client.client.container_name)
            stream = await container_client.download_blob(blob_name)
            data = await stream.readall()
            with open(file_path, "wb") as f:
                f.write(data)

    def download_with_progress(self, blob_name: str, file_path: str, chunk_size=4 * 1024 * 1024):
        """Download with progress bar."""
        container_client = self.client.container_client()
        blob = container_client.get_blob_client(blob_name)
        properties = blob.get_blob_properties()
        total_size = properties.size

        with open(file_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=blob_name) as pbar:
            stream = blob.download_blob()
            for chunk in stream.chunks():
                f.write(chunk)
                pbar.update(len(chunk))
