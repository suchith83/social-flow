# Thread pooling & async execution
"""
Thread Pool Executor for efficient task execution.
"""

import concurrent.futures
from .config import CONFIG


class ThreadPool:
    def __init__(self, min_threads: int = CONFIG.min_threads, max_threads: int = CONFIG.max_threads):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
        self.min_threads = min_threads
        self.max_threads = max_threads

    def submit(self, fn, *args, **kwargs):
        """Submit a task to the thread pool."""
        return self.executor.submit(fn, *args, **kwargs)

    def map(self, fn, iterable):
        """Map a function across an iterable concurrently."""
        return list(self.executor.map(fn, iterable))

    def shutdown(self, wait=True):
        """Shutdown the pool."""
        self.executor.shutdown(wait=wait)
