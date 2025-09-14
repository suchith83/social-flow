# Handles payload compression and decompression
import gzip
import brotli
import zstandard as zstd
from typing import Union


class GzipCompressor:
    def compress(self, data: Union[str, bytes]) -> bytes:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return gzip.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class BrotliCompressor:
    def compress(self, data: Union[str, bytes]) -> bytes:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return brotli.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return brotli.decompress(data)


class ZstdCompressor:
    def __init__(self, level: int = 3):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()

    def compress(self, data: Union[str, bytes]) -> bytes:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.compressor.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self.decompressor.decompress(data)
