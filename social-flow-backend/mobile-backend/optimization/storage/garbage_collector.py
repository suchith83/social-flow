# Cleans up obsolete or expired data
"""
Garbage Collector

Periodically scans metadata and dedup index for orphaned objects and deletes them
from storage backends in safe batches. Supports pluggable deletion strategy and
a dry-run mode for safe testing.

Important: In production, GC must be coordinated with transactional metadata store
so you do not delete objects that became referenced after a scan started.
"""

import threading
import time
from typing import Callable, List
from .config import CONFIG
from .deduplication import Deduplicator

class GarbageCollector:
    def __init__(self, backend_delete: Callable[[str], None], dedup: Deduplicator, run_interval: int = CONFIG.gc_run_interval_sec):
        self.backend_delete = backend_delete
        self.dedup = dedup
        self.run_interval = run_interval
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop_event.wait(self.run_interval):
            try:
                self.run_once(dry_run=True)  # dry-run first
                # In non-demo mode, run actual deletion
                self.run_once(dry_run=False)
            except Exception:
                # Log and continue; in real app, use logging + alerts
                pass

    def run_once(self, dry_run: bool = False, batch_size: int = CONFIG.gc_delete_batch) -> List[str]:
        """
        Inspect dedup index and delete objects whose fingerprint mapping was removed (or zero refs).
        Returns list of deleted keys (if dry_run=False), otherwise returns candidates.
        """
        deleted = []
        # The deduplicator owns the index file and logic. For demo, iterate file.
        # Acquire dedup lock to get stable snapshot
        with self.dedup.lock:
            # fingerprints currently known
            known = set(self.dedup.index.keys())
            # Inverse index: key -> fingerprint
            inv = {}
            for fp, meta in self.dedup.index.items():
                key = meta.get("key")
                inv[key] = fp

        # In a real system, you would query metadata DB for all objects and compare refs.
        # For the demo, scan a simulated storage listing via backend (not available here).
        # We'll provide interface so caller can pass explicit candidates; here we return empty.
        # This function is left deliberately pluggable.
        return deleted

    def stop(self):
        self._stop_event.set()
        self.thread.join(timeout=10)
