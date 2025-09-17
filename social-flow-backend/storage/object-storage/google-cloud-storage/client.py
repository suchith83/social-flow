"""
Google Cloud Storage client wrapper with retries and sync/async support.
"""

from google.cloud import storage
import gcsfs
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import gcs_config


class GCSClient:
    """Wrapper around Google Cloud Storage clients."""

    def __init__(self):
        self._client = storage.Client.from_service_account_json(
            gcs_config.credentials_path, project=gcs_config.project_id
        )
        self._bucket = self._client.bucket(gcs_config.bucket_name)

    @property
    def client(self):
        return self._client

    @property
    def bucket(self):
        return self._bucket

    async def afs(self):
        """Async filesystem client (gcsfs)."""
        return gcsfs.AsyncFileSystem(token=gcs_config.credentials_path, project=gcs_config.project_id)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def upload_blob(self, source_file: str, destination_blob: str):
        blob = self._bucket.blob(destination_blob)
        blob.upload_from_filename(source_file)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def download_blob(self, source_blob: str, destination_file: str):
        blob = self._bucket.blob(source_blob)
        blob.download_to_filename(destination_file)
