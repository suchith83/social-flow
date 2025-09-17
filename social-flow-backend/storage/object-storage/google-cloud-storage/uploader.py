"""
Uploader module for Google Cloud Storage.
"""

import os
import asyncio
from tqdm import tqdm
from .client import GCSClient


class GCSUploader:
    def __init__(self):
        self.client = GCSClient()

    def upload_file(self, file_path: str, blob_name: str):
        """Upload a single file."""
        self.client.upload_blob(file_path, blob_name)

    def upload_directory(self, dir_path: str, prefix: str = ""):
        """Recursively upload a directory to GCS."""
        for root, _, files in os.walk(dir_path):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, dir_path)
                blob_name = os.path.join(prefix, rel_path).replace("\\", "/")
                print(f"Uploading {local_path} -> {blob_name}")
                self.upload_file(local_path, blob_name)

    async def async_upload_file(self, file_path: str, blob_name: str):
        """Async upload with gcsfs."""
        async with await self.client.afs() as fs:
            async with fs.open(f"{self.client.bucket.name}/{blob_name}", "wb") as f:
                with open(file_path, "rb") as src:
                    await f.write(src.read())

    def upload_with_progress(self, file_path: str, blob_name: str, chunk_size=4 * 1024 * 1024):
        """Upload with progress bar."""
        file_size = os.path.getsize(file_path)
        blob = self.client.bucket.blob(blob_name)
        with open(file_path, "rb") as f, tqdm(total=file_size, unit="B", unit_scale=True, desc=file_path) as pbar:
            while chunk := f.read(chunk_size):
                blob.upload_from_string(chunk, content_type="application/octet-stream", client=self.client.client)
                pbar.update(len(chunk))
