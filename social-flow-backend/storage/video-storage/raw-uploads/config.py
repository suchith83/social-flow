"""
Configuration for raw-uploads
"""

import os
from pydantic import BaseSettings


class RawUploadsConfig(BaseSettings):
    ENV: str = os.getenv("ENV", "development")
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024 * 1024))  # 5 GB default
    CHUNK_DIR: str = os.getenv("CHUNK_DIR", "/tmp/raw_uploads/chunks")
    STAGING_DIR: str = os.getenv("STAGING_DIR", "/tmp/raw_uploads/staged")
    ALLOWED_MIME_PREFIXES = ["video/"]
    ALLOWED_EXTENSIONS = [".mp4", ".mov", ".mkv", ".webm", ".avi"]
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "s3")  # s3 or local
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "raw-uploads")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    PRESIGNED_URL_TTL: int = int(os.getenv("PRESIGNED_URL_TTL", 900))  # 15 minutes

config = RawUploadsConfig()
