"""
Utility functions for Azure Blob Storage.
"""

from datetime import datetime, timedelta
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

from .config import azure_blob_config


def generate_presigned_url(blob_name: str, expires_in: int = 3600) -> str:
    """Generate presigned SAS URL for secure temporary access."""
    sas_token = generate_blob_sas(
        account_name=azure_blob_config.connection_string.split(";")[1].split("=")[1],
        container_name=azure_blob_config.container_name,
        blob_name=blob_name,
        account_key=azure_blob_config.connection_string.split(";")[2].split("=")[1],
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(seconds=expires_in),
    )
    return f"https://{azure_blob_config.connection_string.split(';')[1].split('=')[1]}.blob.core.windows.net/{azure_blob_config.container_name}/{blob_name}?{sas_token}"
