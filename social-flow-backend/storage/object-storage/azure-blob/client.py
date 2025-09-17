"""
Azure Blob Storage client wrapper with retry logic and sync/async support.
"""

from azure.storage.blob import BlobServiceClient
from azure.storage.blob.aio import BlobServiceClient as AioBlobServiceClient
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import azure_blob_config


class AzureBlobClient:
    """Wrapper around Azure Blob Storage clients."""

    def __init__(self):
        self._client = BlobServiceClient.from_connection_string(
            azure_blob_config.connection_string
        )

    @property
    def client(self) -> BlobServiceClient:
        """Return sync client."""
        return self._client

    def container_client(self, container_name=None):
        """Get container client."""
        return self._client.get_container_client(
            container_name or azure_blob_config.container_name
        )

    async def aclient(self) -> AioBlobServiceClient:
        """Return async client."""
        return AioBlobServiceClient.from_connection_string(
            azure_blob_config.connection_string
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def upload_blob(self, blob_name: str, data, overwrite=True):
        """Upload blob with retries."""
        container_client = self.container_client()
        container_client.upload_blob(blob_name, data, overwrite=overwrite)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def download_blob(self, blob_name: str) -> bytes:
        """Download blob with retries."""
        container_client = self.container_client()
        blob = container_client.download_blob(blob_name)
        return blob.readall()
