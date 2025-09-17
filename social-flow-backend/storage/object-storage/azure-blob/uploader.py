"""
Uploader module for Azure Blob Storage.
Supports large files, async, and progress bars.
"""

import os
import asyncio
from tqdm import tqdm
from .client import AzureBlobClient


class AzureBlobUploader:
    def __init__(self):
        self.client = AzureBlobClient()

    def upload_file(self, file_path: str, blob_name: str):
        """Upload a single file."""
        with open(file_path, "rb") as f:
            self.client.upload_blob(blob_name, f)

    def upload_directory(self, dir_path: str, prefix: str = ""):
        """Upload a directory recursively."""
        for root, _, files in os.walk(dir_path):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, dir_path)
                blob_name = os.path.join(prefix, rel_path).replace("\\", "/")
                print(f"Uploading {local_path} -> {blob_name}")
                self.upload_file(local_path, blob_name)

    async def async_upload_file(self, file_path: str, blob_name: str):
        """Async upload with progress."""
        async with await self.client.aclient() as client:
            container_client = client.get_container_client(self.client.client.container_name)
            async with aiofiles.open(file_path, "rb") as f:
                await container_client.upload_blob(blob_name, f, overwrite=True)

    def upload_with_progress(self, file_path: str, blob_name: str, chunk_size=4 * 1024 * 1024):
        """Upload with progress bar."""
        file_size = os.path.getsize(file_path)
        container_client = self.client.container_client()
        with open(file_path, "rb") as f, tqdm(total=file_size, unit="B", unit_scale=True, desc=file_path) as pbar:
            while chunk := f.read(chunk_size):
                container_client.upload_blob(blob_name, chunk, overwrite=True)
                pbar.update(len(chunk))
