# Provides content compression utilities
# performance/cdn/optimization/compression.py
"""
Content compression and adaptive encoding.
"""

import gzip
import brotli
import zstandard as zstd
from typing import Tuple
from .utils import logger, content_hash

class Compressor:
    def __init__(self):
        self.algos = {
            "gzip": self._gzip,
            "brotli": self._brotli,
            "zstd": self._zstd
        }

    def compress(self, content: bytes, algo: str = "gzip", level: int = 5) -> Tuple[bytes, str]:
        """Compress using chosen algorithm."""
        if algo not in self.algos:
            raise ValueError(f"Unsupported compression {algo}")
        compressed = self.algos[algo](content, level)
        logger.debug(f"Compressed size={len(compressed)} algo={algo} hash={content_hash(compressed)}")
        return compressed, algo

    def _gzip(self, content: bytes, level: int) -> bytes:
        return gzip.compress(content, compresslevel=level)

    def _brotli(self, content: bytes, level: int) -> bytes:
        return brotli.compress(content, quality=level)

    def _zstd(self, content: bytes, level: int) -> bytes:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(content)
