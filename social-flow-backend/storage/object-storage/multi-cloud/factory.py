"""
Factory for creating storage instances based on provider.
"""

from storage.object-storage.aws-s3.uploader import S3Uploader
from storage.object-storage.aws-s3.downloader import S3Downloader
from storage.object-storage.aws-s3.utils import generate_presigned_url as s3_url

from storage.object-storage.azure-blob.uploader import AzureBlobUploader
from storage.object-storage.azure-blob.downloader import AzureBlobDownloader
from storage.object-storage.azure-blob.utils import generate_presigned_url as azure_url

from storage.object-storage.google-cloud-storage.uploader import GCSUploader
from storage.object-storage.google-cloud-storage.downloader import GCSDownloader
from storage.object-storage.google-cloud-storage.utils import generate_presigned_url as gcs_url

from .base_storage import BaseStorage
from .config import multi_cloud_config


class StorageFactory:
    """Factory for returning correct storage client."""

    @staticmethod
    def get_storage() -> BaseStorage:
        provider = multi_cloud_config.provider.lower()

        if provider == "s3":
            return S3Adapter()
        elif provider == "azure":
            return AzureAdapter()
        elif provider == "gcs":
            return GCSAdapter()
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class S3Adapter(BaseStorage):
    def __init__(self):
        self.uploader = S3Uploader()
        self.downloader = S3Downloader()

    def upload_file(self, file_path: str, object_name: str):
        return self.uploader.upload_file(file_path, object_name)

    def download_file(self, object_name: str, file_path: str):
        return self.downloader.download_file(object_name, file_path)

    async def async_upload_file(self, file_path: str, object_name: str):
        return await self.uploader.async_upload_file(file_path, object_name)

    async def async_download_file(self, object_name: str, file_path: str):
        return await self.downloader.async_download_file(object_name, file_path)

    def generate_presigned_url(self, object_name: str, expires_in=3600):
        from storage.object-storage.aws-s3.client import S3Client
        return s3_url(S3Client().client, object_name, expires_in)


class AzureAdapter(BaseStorage):
    def __init__(self):
        self.uploader = AzureBlobUploader()
        self.downloader = AzureBlobDownloader()

    def upload_file(self, file_path: str, object_name: str):
        return self.uploader.upload_file(file_path, object_name)

    def download_file(self, object_name: str, file_path: str):
        return self.downloader.download_file(object_name, file_path)

    async def async_upload_file(self, file_path: str, object_name: str):
        return await self.uploader.async_upload_file(file_path, object_name)

    async def async_download_file(self, object_name: str, file_path: str):
        return await self.downloader.async_download_file(object_name, file_path)

    def generate_presigned_url(self, object_name: str, expires_in=3600):
        return azure_url(object_name, expires_in)


class GCSAdapter(BaseStorage):
    def __init__(self):
        self.uploader = GCSUploader()
        self.downloader = GCSDownloader()

    def upload_file(self, file_path: str, object_name: str):
        return self.uploader.upload_file(file_path, object_name)

    def download_file(self, object_name: str, file_path: str):
        return self.downloader.download_file(object_name, file_path)

    async def async_upload_file(self, file_path: str, object_name: str):
        return await self.uploader.async_upload_file(file_path, object_name)

    async def async_download_file(self, object_name: str, file_path: str):
        return await self.downloader.async_download_file(object_name, file_path)

    def generate_presigned_url(self, object_name: str, expires_in=3600):
        return gcs_url(object_name, expires_in)
