# Manages multiple storage tiers (hot/warm/cold)
"""
TieredStorage

Coordinates hot, warm and cold backends and lifecycle transitions.
- Hot: low-latency local disk (for frequently accessed objects)
- Warm: remote object store (S3-like)
- Cold: archive backend

Features:
 - Put/get semantics with tier-aware retrieval and automatic tier promotion on access.
 - Lifecycle transitions based on age/size (hot->warm->cold).
 - Transparent decompression on get if object was stored compressed.
 - Integration points for metrics and GC.

This implementation is synchronous for clarity; it can be layered over AsyncWriter for writes.
"""

import time
from typing import Optional, Dict, Any
from .config import CONFIG
from .storage_backends import LocalDiskBackend, S3Backend, ColdArchiveBackend
from .compression_adapter import CompressionAdapter
from .metrics import StorageMetrics

class TieredStorage:
    def __init__(self, hot_backend: LocalDiskBackend, warm_backend: S3Backend, cold_backend: ColdArchiveBackend):
        self.hot = hot_backend
        self.warm = warm_backend
        self.cold = cold_backend
        self.compressor = CompressionAdapter()
        self.metrics = StorageMetrics()
        # metadata store simulated via small in-memory dict
        # metadata: key -> {"created": ts, "size": int, "tier": "hot/warm/cold", "compression": "zlib|lzma|brotli|identity", "fingerprint": str}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def put(self, key: str, data: bytes, metadata: Dict[str, Any] = None, prefer_compress: Optional[str] = None) -> None:
        """
        Put object into appropriate initial tier:
         - small objects -> hot
         - medium -> warm
         - very large -> warm or cold depending on policy
        """
        size = len(data)
        meta = metadata.copy() if metadata else {}
        # choose initial tier
        if size <= CONFIG.hot_tier_max_bytes:
            backend = self.hot
            tier = "hot"
        elif size <= CONFIG.warm_tier_max_bytes:
            backend = self.warm
            tier = "warm"
        else:
            backend = self.warm  # large objects initially land warm; lifecycle may move to cold
            tier = "warm"

        # compress if beneficial
        compressed, alg = self.compressor.compress(data, prefer_compress)
        meta["compression"] = alg
        # write
        t0 = time.time()
        backend.put(key, compressed, meta)
        duration = time.time() - t0
        self.metrics.log("put_latency", duration)
        self.metadata[key] = {"created": time.time(), "size": size, "tier": tier, "compression": alg, "fingerprint": meta.get("fingerprint")}
        self.metrics.incr("objects_put", 1)
        if alg != "identity":
            self.metrics.incr("bytes_saved", size - len(compressed))

    def _get_from_backend(self, backend, key: str) -> Optional[bytes]:
        t0 = time.time()
        data = backend.get(key)
        self.metrics.log("get_latency", time.time() - t0)
        return data

    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve object, trying hot -> warm -> cold. If found in lower tier, promote to hot.
        Decompress before returning.
        """
        # check metadata to know expected tier if available
        meta = self.metadata.get(key)
        # Try hot
        data = self.hot.get(key)
        if data is not None:
            self.metrics.incr("hot_hits", 1)
            compression = meta.get("compression") if meta else "identity"
            try:
                return self.compressor.decompress(data, compression)
            except Exception:
                # If decompress fails, return raw data as fallback
                return data

        # Try warm
        data = self.warm.get(key)
        if data is not None:
            self.metrics.incr("warm_hits", 1)
            # promote asynchronously or synchronously - here synchronous promote for simplicity
            self.hot.put(key, data, {"promoted_from": "warm"})
            if meta:
                meta["tier"] = "hot"
            compression = meta.get("compression") if meta else "identity"
            try:
                return self.compressor.decompress(data, compression)
            except Exception:
                return data

        # Try cold
        data = self.cold.get(key)
        if data is not None:
            self.metrics.incr("cold_hits", 1)
            # promotion to warm might be required (expensive)
            self.warm.put(key, data, {"promoted_from": "cold"})
            if meta:
                meta["tier"] = "warm"
            compression = meta.get("compression") if meta else "identity"
            try:
                return self.compressor.decompress(data, compression)
            except Exception:
                return data

        self.metrics.incr("misses", 1)
        return None

    def delete(self, key: str) -> None:
        """
        Delete object from all tiers and metadata. Coordination with dedup is required externally.
        """
        for b in (self.hot, self.warm, self.cold):
            try:
                b.delete(key)
            except Exception:
                pass
        if key in self.metadata:
            del self.metadata[key]
        self.metrics.incr("objects_deleted", 1)

    def lifecycle_tick(self):
        """
        Run periodic lifecycle transitions according to metadata timestamps.
        Move objects older than thresholds from hot->warm and warm->cold.
        In prod, this should be a scheduled job reading from metadata DB.
        """
        now = time.time()
        keys = list(self.metadata.keys())
        for key in keys:
            info = self.metadata[key]
            age_days = (now - info["created"]) / (60 * 60 * 24)
            try:
                if info["tier"] == "hot" and age_days >= CONFIG.hot_to_warm_days:
                    # move to warm (copy then delete hot)
                    data = self.hot.get(key)
                    if data is not None:
                        self.warm.put(key, data, {"migrated_from": "hot"})
                        self.hot.delete(key)
                        info["tier"] = "warm"
                        self.metrics.incr("migrations_hot_to_warm", 1)
                elif info["tier"] == "warm" and age_days >= CONFIG.warm_to_cold_days:
                    data = self.warm.get(key)
                    if data is not None:
                        self.cold.put(key, data, {"migrated_from": "warm"})
                        self.warm.delete(key)
                        info["tier"] = "cold"
                        self.metrics.incr("migrations_warm_to_cold", 1)
            except Exception:
                # in production we would capture and alert
                continue

    def stats(self):
        """
        Return rough counts and sizes by tier based on metadata.
        """
        counts = {"hot": 0, "warm": 0, "cold": 0}
        bytes_by_tier = {"hot": 0, "warm": 0, "cold": 0}
        for key, info in self.metadata.items():
            tier = info.get("tier", "warm")
            counts[tier] = counts.get(tier, 0) + 1
            bytes_by_tier[tier] = bytes_by_tier.get(tier, 0) + info.get("size", 0)
        return {"counts": counts, "bytes": bytes_by_tier}
