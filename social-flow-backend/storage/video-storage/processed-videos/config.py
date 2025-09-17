"""
Configuration for processed video pipeline
"""

import os
from pydantic import BaseSettings


class ProcessedVideosConfig(BaseSettings):
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "s3")
    STORAGE_BUCKET: str = os.getenv("STORAGE_BUCKET", "processed-videos")
    STORAGE_ENDPOINT: str = os.getenv("STORAGE_ENDPOINT", "http://localhost:9000")
    STORAGE_ACCESS_KEY: str = os.getenv("STORAGE_ACCESS_KEY", "minioadmin")
    STORAGE_SECRET_KEY: str = os.getenv("STORAGE_SECRET_KEY", "minioadmin")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/tmp/processed-videos")


config = ProcessedVideosConfig()
