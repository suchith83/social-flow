# Orchestrates storage operations and policies
"""
StorageManager - high-level orchestrator for storage optimization.

Combines:
 - TieredStorage for tier transitions and get/put semantics.
 - AsyncWriter for buffered writes.
 - Deduplicator for content-addressing and refcounts.
 - GarbageCollector for cleanup.
 - Metrics aggregation.

Usage:
   backends = (LocalDiskBackend('./hot'), S3Backend('mybucket'), ColdArchiveBackend('./cold'))
   ts = TieredStorage(*backends)
   manager = StorageManager(tiered=ts)
   manager.put('user/123/thumbnail', b'...')  # uses async writer + dedup/compression
   data = manager.get('user/123/thumbnail')
"""

import threading
from typing import Optional, Dict, Any
from .tiered_storage import TieredStorage
from .async_writer import AsyncWriter
from .deduplication import Deduplicator
from .garbage_collector import GarbageCollector
from .metrics import StorageMetrics

class StorageManager:
    def __init__(self, tiered: TieredStorage, use_async_writer: bool = True):
        self.tiered = tiered
        self.metrics = StorageMetrics()
        self.dedup = Deduplicator()
        # wire async writer to tiered.put (synchronous backend write wrapper)
        if use_async_writer:
            # backend_put expects (key, data, meta)
            def backend_put(key, data, meta):
                # Directly use tiered.hot/warm decision inside underlying put by delegating to tiered.put
                # But tiered.put expects raw uncompressed original data; to avoid double compression,
                # we will call underlying backend directly. For simplicity, write to hot always in async writer.
                self.tiered.hot.put(key, data, meta)
            self.writer = AsyncWriter(backend_put)
        else:
            self.writer = None

        # GC that uses tiered.delete; in production GC logic should be integrated with dedup
        self.gc = GarbageCollector(self.tiered.delete, self.dedup)

        # start a lifecycle thread to promote/demote tiers periodically
        self._stop_event = threading.Event()
        self._lifecycle_thread = threading.Thread(target=self._lifecycle_loop, daemon=True)
        self._lifecycle_thread.start()

    def put(self, key: str, data: bytes, metadata: Dict[str, Any] = None, prefer_compress: Optional[str] = None) -> bool:
        """
        High-level put: first attempt dedup; if unique, submit to async writer (if configured),
        otherwise increment refcount and update metadata.
        """
        fp, existed, canonical = self.dedup.ensure_unique(data, suggested_key=key)
        if existed:
            # update metadata store to point to canonical and increment refcount further
            self.tiered.metadata[canonical] = self.tiered.metadata.get(canonical, {})
            # in real app, update DB references
            self.metrics.incr("dedup_hits", 1)
            return True

        # not found -> write
        if self.writer:
            enqueued = self.writer.submit(key, data, metadata or {}, prefer_compress)
            if not enqueued:
                # fallback to synchronous put if queue full
                self.tiered.put(key, data, metadata or {}, prefer_compress)
                self.metrics.incr("writes_sync_fallback", 1)
                return True
            self.metrics.incr("writes_async", 1)
            return True
        else:
            self.tiered.put(key, data, metadata or {}, prefer_compress)
            self.metrics.incr("writes_sync", 1)
            return True

    def get(self, key: str) -> Optional[bytes]:
        return self.tiered.get(key)

    def delete(self, key: str) -> None:
        # Decrement dedup ref, and if zero, delete actual storage
        meta = self.tiered.metadata.get(key, {})
        fp = meta.get("fingerprint")
        if fp:
            remaining = self.dedup.unref(fp)
            if remaining <= 0:
                self.tiered.delete(key)
        else:
            self.tiered.delete(key)

    def flush_metrics(self):
        report = self.metrics.flush()
        # include tiered metrics too
        report.update(self.tiered.metrics.flush())
        return report

    def shutdown(self):
        # stop lifecycle thread, writer and GC gracefully
        self._stop_event.set()
        if self.writer:
            self.writer.shutdown()
        if self.gc:
            self.gc.stop()

    def _lifecycle_loop(self):
        # run lifecycle_tick periodically (every minute)
        while not self._stop_event.wait(60):
            try:
                self.tiered.lifecycle_tick()
            except Exception:
                pass
