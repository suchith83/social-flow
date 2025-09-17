"""
Configuration module for Google Cloud Storage integration.
"""

from pydantic import BaseSettings, Field


class GCSConfig(BaseSettings):
    project_id: str = Field(..., env="GCP_PROJECT_ID")
    bucket_name: str = Field(..., env="GCS_BUCKET_NAME")
    credentials_path: str = Field(..., env="GOOGLE_APPLICATION_CREDENTIALS")

    class Config:
        env_file = ".env"


gcs_config = GCSConfig()
