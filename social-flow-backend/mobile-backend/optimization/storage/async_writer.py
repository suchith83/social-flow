# Async writing of data to storage
"""
AsyncWriter

Provides an asynchronous, batched writer pipeline for putting objects into storage.
- Buffers writes to allow batching (useful for many small objects).
- Coordinates with deduplication and compression adapters.
- Supports backpressure and graceful shutdown.

This implementation uses threading + background worker to avoid adding an extra runtime
requirement (no asyncio). In production you may switch to an asyncio-based pipeline.
"""

import threading
import time
from queue import Queue, Full, Empty
from typing import Callable, Dict, Any, Optional, Tuple
from .config import CONFIG
from .compression_adapter import CompressionAdapter
from .deduplication import Deduplicator
from .storage_backends import LocalDiskBackend, S3Backend

class AsyncWriter:
    def __init__(self, backend_put: Callable[[str, bytes, dict], None], queue_max: int = CONFIG.write_queue_max):
        self.queue: Queue = Queue(maxsize=queue_max)
        self.backend_put = backend_put
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.compressor = CompressionAdapter()
        self.dedup = Deduplicator()

    def submit(self, key: str, data: bytes, metadata: Dict[str, Any] = None, prefer_compress: Optional[str] = None) -> bool:
        """Submit a write. Returns True if enqueued, False if rejected due to full queue."""
        try:
            entry = {"key": key, "data": data, "meta": metadata or {}, "prefer_compress": prefer_compress}
            self.queue.put_nowait(entry)
            return True
        except Full:
            return False  # caller can retry or fall back to direct write

    def _worker(self):
        """
        Consume queue, batch small writes to reduce backend call overhead, and apply dedup/compression.
        """
        batch = []
        last_flush = time.time()
        while not self._stop_event.is_set():
            try:
                entry = self.queue.get(timeout=CONFIG.write_flush_interval_sec)
                batch.append(entry)
                # flush conditions: batch size or time elapsed
                if len(batch) >= CONFIG.write_batch_size or (time.time() - last_flush) >= CONFIG.write_flush_interval_sec:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
            except Empty:
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                continue

        # drain remaining
        while True:
            try:
                entry = self.queue.get_nowait()
                batch.append(entry)
            except Empty:
                break
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch):
        """
        Process each entry in batch:
         1. Deduplicate: compute fingerprint; if exists, update refs and skip actual put.
         2. Compress (if beneficial).
         3. Put to backend.
        """
        for entry in batch:
            key = entry["key"]
            data = entry["data"]
            meta = entry["meta"]
            prefer = entry.get("prefer_compress")
            # deduplication path
            fp, existed, canonical = self.dedup.ensure_unique(data, suggested_key=key)
            meta = meta.copy()
            meta["fingerprint"] = fp
            if existed:
                # if already exists, nothing to write; metadata may still need to be updated
                meta["dedup_hit"] = True
                # Optionally update a metadata DB or notify; here we just continue
                continue
            # compress if beneficial
            compressed_data, alg = self.compressor.compress(data, prefer)
            meta["compression"] = alg
            # write to backend (synchronous call)
            try:
                self.backend_put(key, compressed_data, meta)
            except Exception:
                # On failure, decrement dedup ref to avoid leakage
                self.dedup.unref(fp)
                # In production, implement retry with backoff + persistent failure queue
                continue

    def shutdown(self, timeout: Optional[float] = None):
        """Signal worker to stop and wait for drain."""
        self._stop_event.set()
        self.thread.join(timeout)
