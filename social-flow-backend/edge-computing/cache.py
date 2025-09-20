import threading
import time
from typing import Any, Dict, Optional


class InMemoryCache:
    """Simple thread-safe in-memory cache with TTL."""

    def __init__(self, default_ttl: int = 60):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl
        self._stop = False
        self._cleaner = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleaner.start()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expire = time.time() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            self._store[key] = {"value": value, "expire": expire}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            if item["expire"] < time.time():
                # expired
                del self._store[key]
                return None
            return item["value"]

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _cleanup_loop(self) -> None:
        while not self._stop:
            now = time.time()
            with self._lock:
                keys = [k for k, v in self._store.items() if v["expire"] < now]
                for k in keys:
                    del self._store[k]
            time.sleep(max(1, int(self.default_ttl / 5)))

    def stop(self) -> None:
        self._stop = True
        if self._cleaner.is_alive():
            self._cleaner.join(timeout=1)
