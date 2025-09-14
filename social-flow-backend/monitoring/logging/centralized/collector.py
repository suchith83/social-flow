# Collects logs from different sources
# monitoring/logging/centralized/collector.py
"""
Log collector for centralized logging.
Collects from files, API, or streaming sources.
"""

import time
from .utils import batch_logs, normalize_log
from .security import mask_sensitive


class LogCollector:
    def __init__(self, storage, indexer, batch_size=100):
        self.storage = storage
        self.indexer = indexer
        self.batch_size = batch_size
        self.buffer = []

    def collect(self, log: dict):
        """Collect single log into buffer."""
        norm = normalize_log(mask_sensitive(log))
        self.buffer.append(norm)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """Flush buffered logs to storage and indexer."""
        if not self.buffer:
            return
        self.storage.store(self.buffer)
        for batch in batch_logs(self.buffer, self.batch_size):
            self.indexer.index(batch)
        self.buffer.clear()

    def collect_stream(self, stream, interval=5):
        """Continuously collect logs from a stream generator."""
        for log in stream:
            self.collect(log)
            time.sleep(interval)
