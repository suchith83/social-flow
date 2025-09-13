# Profiles runtime performance
"""
Runtime Profiler
Measures latency, throughput, and task cost.
"""

import time


class Profiler:
    def __init__(self):
        self.records = []

    def profile(self, fn, *args, **kwargs):
        """Profile function execution time."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration = time.perf_counter() - start
        self.records.append(duration)
        return result, duration

    def avg_latency(self) -> float:
        return sum(self.records) / len(self.records) if self.records else 0.0
