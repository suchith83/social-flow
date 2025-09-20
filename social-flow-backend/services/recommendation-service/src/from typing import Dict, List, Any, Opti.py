from typing import Dict, List, Any, Optional
from threading import Lock
import time


class InMemoryMetricsStore:
    """Simple thread-safe in-memory metrics store for development and tests."""

    def __init__(self):
        self._lock = Lock()
        # store: name -> list of points (dict with value, tags, timestamp)
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def add_point(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        point = {"value": float(value), "tags": tags or {}, "timestamp": float(timestamp)}
        with self._lock:
            self._store.setdefault(name, []).append(point)

    def query(self, name: str, start_ts: Optional[float] = None, end_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        with self._lock:
            points = list(self._store.get(name, []))
        if start_ts is not None:
            points = [p for p in points if p["timestamp"] >= start_ts]
        if end_ts is not None:
            points = [p for p in points if p["timestamp"] <= end_ts]
        # return sorted by timestamp ascending
        return sorted(points, key=lambda p: p["timestamp"])
