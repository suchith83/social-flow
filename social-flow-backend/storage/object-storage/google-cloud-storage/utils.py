"""
Utility functions for GCS integration.
"""

from datetime import timedelta
from .client import GCSClient


def generate_presigned_url(blob_name: str, expires_in: int = 3600) -> str:
    """Generate signed URL for secure temporary access."""
    client = GCSClient()
    blob = client.bucket.blob(blob_name)
    url = blob.generate_signed_url(expiration=timedelta(seconds=expires_in))
    return url
