# batch_invalidator.py
# Created by Create-Invalidation.ps1
"""
batch_invalidator.py
--------------------
High-level helper to batch invalidations and apply across multiple backends:
- immediate deletion in Redis / Memcached
- publish invalidation events to pubsub
- trigger CDN invalidation for affected paths
- configurable concurrency / batching
"""

from __future__ import annotations
import time
import logging
from typing import Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BatchInvalidator:
    """
    Compose multiple invalidation backends into a single API.

    backends: dict with optional keys 'redis', 'pubsub', 'cdn' containing invalidator objects
    batch_size: how many keys per batch for backend.delete operations
    concurrency: thread pool size for parallel backends
    """

    def __init__(self, backends: dict, batch_size: int = 500, concurrency: int = 4):
        self.backends = backends
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)

    @staticmethod
    def _chunks(iterable: Iterable[str], size: int):
        """Yield successive chunks from iterable."""
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _invalidate_chunk(self, backend_name: str, invalidator, chunk: List[str]):
        """Call appropriate method on backend; catches and logs backend-specific errors."""
        try:
            logger.debug("Invalidating %d keys on backend=%s", len(chunk), backend_name)
            # Prefer direct invalidate; fall back to publish if only pubsub exists
            if hasattr(invalidator, "invalidate"):
                invalidator.invalidate(chunk)
            elif hasattr(invalidator, "invalidate_keys"):
                invalidator.invalidate_keys(chunk)
            else:
                # try a publish-like interface
                try:
                    invalidator.publish({"cmd": "invalidate", "keys": chunk})
                except Exception as e:
                    logger.exception("Backend %s could not be invoked", backend_name)
                    raise
            return True
        except Exception as e:
            logger.exception("Error invalidating on backend %s: %s", backend_name, e)
            return False

    def invalidate(self, keys: Iterable[str], cdn_paths: Optional[Iterable[str]] = None):
        """
        Invalidate keys across configured backends in batches and in parallel.

        Steps:
        1. Batch-delete on Redis-like backends.
        2. Publish invalidation events to pubsub.
        3. Trigger CDN invalidation for paths (if provided).
        """
        keys = list(keys)
        if not keys and not cdn_paths:
            return

        tasks = []

        with ThreadPoolExecutor(max_workers=self.concurrency) as exe:
            # 1) Redis-like backends
            redis_backends = {k: v for k, v in self.backends.items() if k in ("redis", "memcached")}
            for backend_name, backend in redis_backends.items():
                for chunk in self._chunks(keys, self.batch_size):
                    tasks.append(exe.submit(self._invalidate_chunk, backend_name, backend, chunk))

            # 2) pubsub
            if "pubsub" in self.backends:
                pubsub = self.backends["pubsub"]
                # pubsub can take whole key list at once
                tasks.append(exe.submit(self._invalidate_chunk, "pubsub", pubsub, keys))

            # 3) CDN
            if "cdn" in self.backends and cdn_paths:
                cdn = self.backends["cdn"]
                # cdn typically expects path strings; ensure list
                cdn_paths_list = list(cdn_paths)
                # send paths in batches too
                for chunk in self._chunks(cdn_paths_list, self.batch_size):
                    tasks.append(exe.submit(self._invalidate_chunk, "cdn", cdn, chunk))

            # wait for completions and aggregate results
            results = [f.result() for f in as_completed(tasks)]
            success_count = sum(1 for r in results if r)
            logger.info("Batch invalidation completed: %d/%d tasks successful", success_count, len(results))

        return success_count, len(results)
