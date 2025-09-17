"""
Configuration module for Azure Blob Storage.
"""

from pydantic import BaseSettings, Field


class AzureBlobConfig(BaseSettings):
    connection_string: str = Field(..., env="AZURE_STORAGE_CONNECTION_STRING")
    container_name: str = Field(..., env="AZURE_BLOB_CONTAINER")

    class Config:
        env_file = ".env"


azure_blob_config = AzureBlobConfig()
