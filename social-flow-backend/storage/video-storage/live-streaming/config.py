"""
Configuration for Live Streaming
"""

import os
from pydantic import BaseSettings


class LiveStreamingConfig(BaseSettings):
    ENV: str = os.getenv("ENV", "development")
    RTMP_PORT: int = int(os.getenv("RTMP_PORT", "1935"))
    HLS_OUTPUT_DIR: str = os.getenv("HLS_OUTPUT_DIR", "/tmp/hls")
    DASH_OUTPUT_DIR: str = os.getenv("DASH_OUTPUT_DIR", "/tmp/dash")
    TRANSCODER_PATH: str = os.getenv("TRANSCODER_PATH", "ffmpeg")
    WS_HOST: str = os.getenv("WS_HOST", "0.0.0.0")
    WS_PORT: int = int(os.getenv("WS_PORT", "8080"))
    CDN_URL: str = os.getenv("CDN_URL", "http://cdn.local/live")


config = LiveStreamingConfig()
