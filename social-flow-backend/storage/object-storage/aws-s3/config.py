"""
Configuration module for AWS S3 integration.
Provides typed settings and environment-based configuration loading.
"""

import os
from pydantic import BaseSettings, Field


class S3Config(BaseSettings):
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field("us-east-1", env="AWS_REGION")
    bucket_name: str = Field(..., env="AWS_S3_BUCKET")

    class Config:
        env_file = ".env"


# Load once and reuse
s3_config = S3Config()
