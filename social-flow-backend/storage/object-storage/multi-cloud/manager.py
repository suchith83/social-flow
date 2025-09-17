"""
Multi-cloud storage manager: high-level abstraction layer.
"""

from .factory import StorageFactory


class MultiCloudStorageManager:
    """High-level API to manage uploads/downloads across providers."""

    def __init__(self):
        self.backend = StorageFactory.get_storage()

    def upload_file(self, file_path: str, object_name: str):
        return self.backend.upload_file(file_path, object_name)

    def download_file(self, object_name: str, file_path: str):
        return self.backend.download_file(object_name, file_path)

    async def async_upload_file(self, file_path: str, object_name: str):
        return await self.backend.async_upload_file(file_path, object_name)

    async def async_download_file(self, object_name: str, file_path: str):
        return await self.backend.async_download_file(object_name, file_path)

    def generate_presigned_url(self, object_name: str, expires_in=3600):
        return self.backend.generate_presigned_url(object_name, expires_in)
