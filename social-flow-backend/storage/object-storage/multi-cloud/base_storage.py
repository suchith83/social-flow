"""
Base abstract class for multi-cloud storage.
"""

import abc


class BaseStorage(abc.ABC):
    """Abstract base class for cloud storage providers."""

    @abc.abstractmethod
    def upload_file(self, file_path: str, object_name: str):
        pass

    @abc.abstractmethod
    def download_file(self, object_name: str, file_path: str):
        pass

    @abc.abstractmethod
    async def async_upload_file(self, file_path: str, object_name: str):
        pass

    @abc.abstractmethod
    async def async_download_file(self, object_name: str, file_path: str):
        pass

    @abc.abstractmethod
    def generate_presigned_url(self, object_name: str, expires_in: int = 3600) -> str:
        pass
