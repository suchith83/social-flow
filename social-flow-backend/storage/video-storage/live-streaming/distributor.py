"""
Distributor for delivering live streams via CDN
"""

from .config import config
from .utils import logger


class Distributor:
    def __init__(self):
        self.cdn_url = config.CDN_URL

    def get_stream_url(self, stream_id: str, protocol: str = "hls") -> str:
        if protocol == "hls":
            return f"{self.cdn_url}/{stream_id}/index.m3u8"
        elif protocol == "dash":
            return f"{self.cdn_url}/{stream_id}/manifest.mpd"
        else:
            logger.error("Unsupported protocol")
            return ""
