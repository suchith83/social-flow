"""
Configuration for Multi-Cloud Storage.
"""

from pydantic import BaseSettings, Field


class MultiCloudConfig(BaseSettings):
    provider: str = Field(..., env="CLOUD_PROVIDER")  # s3 | azure | gcs

    class Config:
        env_file = ".env"


multi_cloud_config = MultiCloudConfig()
