# Batches small network calls to save bandwidth
"""
Request Batching Module
Aggregates multiple lightweight requests into fewer network calls.
"""

import time
import threading
from queue import Queue
from typing import Any, Dict, List
from .config import CONFIG


class RequestBatcher:
    def __init__(self):
        self.queue = Queue()
        self.lock = threading.Lock()
        self.last_flush_time = time.time()
        self.batches: List[List[Dict[str, Any]]] = []

    def add_request(self, request: Dict[str, Any]) -> None:
        """Add a request to the batching queue."""
        self.queue.put(request)
        if self.queue.qsize() >= CONFIG.max_batch_size:
            self.flush()

    def flush(self) -> List[Dict[str, Any]]:
        """Flush the batch into a single network call payload."""
        with self.lock:
            batch = []
            while not self.queue.empty() and len(batch) < CONFIG.max_batch_size:
                batch.append(self.queue.get())
            if batch:
                self.batches.append(batch)
            self.last_flush_time = time.time()
            return batch

    def periodic_flush(self):
        """Flush periodically based on batch window."""
        while True:
            if (time.time() - self.last_flush_time) * 1000 >= CONFIG.batch_window_ms:
                self.flush()
            time.sleep(CONFIG.batch_window_ms / 1000)
