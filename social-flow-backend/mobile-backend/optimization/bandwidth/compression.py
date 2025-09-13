# Handles data compression/decompression
"""
Compression Engine for Bandwidth Optimization
Supports multi-algorithm compression for efficient transmission.
"""

import zlib
import lzma
import brotli
from .config import CONFIG


class CompressionEngine:
    def __init__(self, level: int = CONFIG.compression_level):
        self.level = level

    def compress(self, data: bytes, algorithm: str = "zlib") -> bytes:
        """Compress data using the specified algorithm."""
        if len(data) < CONFIG.min_compression_size:
            return data  # Skip compression for very small payloads

        if algorithm == "zlib":
            return zlib.compress(data, self.level)
        elif algorithm == "lzma":
            return lzma.compress(data, preset=self.level)
        elif algorithm == "brotli":
            return brotli.compress(data, quality=self.level)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def decompress(self, data: bytes, algorithm: str = "zlib") -> bytes:
        """Decompress data using the specified algorithm."""
        if algorithm == "zlib":
            return zlib.decompress(data)
        elif algorithm == "lzma":
            return lzma.decompress(data)
        elif algorithm == "brotli":
            return brotli.decompress(data)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
