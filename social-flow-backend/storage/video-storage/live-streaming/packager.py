"""
HLS/DASH packaging utilities
"""

import os
from .utils import ensure_dir, logger
from .config import config


class Packager:
    def __init__(self):
        ensure_dir(config.HLS_OUTPUT_DIR)
        ensure_dir(config.DASH_OUTPUT_DIR)

    def get_hls_playlist(self, stream_id: str) -> str:
        path = os.path.join(config.HLS_OUTPUT_DIR, stream_id, "index.m3u8")
        if os.path.exists(path):
            return path
        logger.warning(f"HLS playlist not found for {stream_id}")
        return ""

    def get_dash_manifest(self, stream_id: str) -> str:
        path = os.path.join(config.DASH_OUTPUT_DIR, stream_id, "manifest.mpd")
        if os.path.exists(path):
            return path
        logger.warning(f"DASH manifest not found for {stream_id}")
        return ""
