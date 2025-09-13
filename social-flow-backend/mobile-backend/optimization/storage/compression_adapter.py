# Adapts data compression/decompression algorithms
"""
Adaptive Compression Adapter

Wraps multiple compression algorithms and chooses strategy based on object size,
compressibility heuristics, and backend capabilities. Provides utilities to
compress/decompress and to store metadata that indicates compression used.

Note: uses Python standard libs (zlib, lzma) and optional brotli if available.
"""

import zlib
import lzma
import hashlib
from typing import Tuple, Optional
from .config import CONFIG

try:
    import brotli  # type: ignore
    _HAS_BROTLI = True
except Exception:
    _HAS_BROTLI = False

class CompressionAdapter:
    def __init__(self, default_algorithm: str = CONFIG.default_algorithm, level: int = CONFIG.compression_level):
        self.default = default_algorithm
        self.level = level

    def _compress_zlib(self, data: bytes) -> bytes:
        return zlib.compress(data, self.level)

    def _decompress_zlib(self, data: bytes) -> bytes:
        return zlib.decompress(data)

    def _compress_lzma(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=self.level)

    def _decompress_lzma(self, data: bytes) -> bytes:
        return lzma.decompress(data)

    def _compress_brotli(self, data: bytes) -> bytes:
        if not _HAS_BROTLI:
            raise RuntimeError("brotli not available")
        return brotli.compress(data, quality=self.level)

    def _decompress_brotli(self, data: bytes) -> bytes:
        if not _HAS_BROTLI:
            raise RuntimeError("brotli not available")
        return brotli.decompress(data)

    def guess_compressible(self, data: bytes) -> bool:
        """
        Heuristic: sample a prefix and check for entropy via hashing collisions.
        Very small objects or already compressed content will not compress well.
        """
        if len(data) < CONFIG.dedup_min_size:
            return False
        sample = data[:1024]
        # Rough heuristic: count repeated bytes
        unique = len(set(sample))
        density = unique / len(sample)
        # Lower density -> likely compressible
        return density < 0.8

    def compress(self, data: bytes, prefer: Optional[str] = None) -> Tuple[bytes, str]:
        """
        Compress data and return (compressed_bytes, algorithm_used)
        Chooses algorithm based on prefer/default and availability.
        """
        alg = prefer or self.default
        # If object is unlikely compressible, skip compression
        if not self.guess_compressible(data):
            return data, "identity"

        if alg == "brotli" and _HAS_BROTLI:
            try:
                return self._compress_brotli(data), "brotli"
            except Exception:
                pass  # fallback
        if alg == "lzma":
            try:
                return self._compress_lzma(data), "lzma"
            except Exception:
                pass
        # fallback to zlib
        return self._compress_zlib(data), "zlib"

    def decompress(self, data: bytes, algorithm: str) -> bytes:
        if algorithm == "identity":
            return data
        if algorithm == "zlib":
            return self._decompress_zlib(data)
        if algorithm == "lzma":
            return self._decompress_lzma(data)
        if algorithm == "brotli":
            return self._decompress_brotli(data)
        raise ValueError(f"Unknown compression algorithm: {algorithm}")

    def fingerprint(self, data: bytes) -> str:
        """Stable fingerprint used for deduplication (sha256 hex)."""
        return hashlib.sha256(data).hexdigest()
