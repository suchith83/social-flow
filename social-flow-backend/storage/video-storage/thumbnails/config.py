"""
Configuration for thumbnails module
"""

import os
from pydantic import BaseSettings


class ThumbnailsConfig(BaseSettings):
    ENV: str = os.getenv("ENV", "development")
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    FFPROBE_PATH: str = os.getenv("FFPROBE_PATH", "ffprobe")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/tmp/thumbnails")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "s3")  # s3 or local
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "video-thumbnails")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    PRESIGNED_URL_TTL: int = int(os.getenv("PRESIGNED_URL_TTL", "3600"))
    DEFAULT_SIZES: str = os.getenv("DEFAULT_SIZES", "320x180,640x360,1280x720")
    ALLOW_SPRITE: bool = os.getenv("ALLOW_SPRITE", "true").lower() in ("1","true","yes")
    ENABLE_PHASH: bool = os.getenv("ENABLE_PHASH", "true").lower() in ("1","true","yes")
    MAX_THUMBNAILS: int = int(os.getenv("MAX_THUMBNAILS", "10"))
    # Webhook callback when thumbnails are ready
    CALLBACK_URL: str = os.getenv("CALLBACK_URL", "")


config = ThumbnailsConfig()
