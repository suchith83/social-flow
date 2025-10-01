# metrics.py
import time
import threading
from collections import defaultdict
from typing import Dict, Any, Callable


class MetricsCollector:
    """
    Collects metrics (counters, gauges, histograms).
    Thread-safe, suitable for high-throughput systems.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)

    def increment(self, name: str, value: int = 1):
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float):
        """Record a value in a histogram."""
        with self._lock:
            self._histograms[name].append(value)

    def measure_duration(self, name: str) -> Callable:
        """Decorator to measure function execution time into a histogram."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.time() - start
                    self.observe(name, duration)
            return wrapper
        return decorator

    def export(self) -> Dict[str, Any]:
        """Export current metrics snapshot."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: list(v) for k, v in self._histograms.items()}
            }
