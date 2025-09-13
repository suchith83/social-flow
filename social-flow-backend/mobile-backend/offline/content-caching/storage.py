# Cache storage abstraction (Redis, local FS, S3)
"""
Storage abstraction for cached content:
- Redis for metadata & quick presence checks
- Disk storage for binary blobs with atomic writes
- Fallback logic and helpers for eviction and size accounting
"""

import os
import io
import shutil
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple
from .config import get_config
import logging

logger = logging.getLogger("content_caching.storage")
config = get_config()

try:
    import redis
except Exception:
    redis = None
    logger.warning("redis-py not installed or unavailable; Redis caching disabled.")


class StorageError(Exception):
    pass


class RedisStore:
    """
    Small wrapper around Redis for storing metadata & small blobs.
    """
    def __init__(self, url: str = None):
        if not redis:
            raise StorageError("redis module not available")
        self.client = redis.from_url(url or config.REDIS_URL)

    def set_meta(self, key: str, value: dict, ttl: Optional[int] = None):
        k = f"cache_meta:{key}"
        self.client.set(k, str(value))
        if ttl:
            self.client.expire(k, ttl)

    def get_meta(self, key: str) -> Optional[dict]:
        k = f"cache_meta:{key}"
        v = self.client.get(k)
        if v is None:
            return None
        try:
            # stored as str(dict) -> eval safe-ish but better to use json; keep simple for now
            return eval(v)
        except Exception:
            return None

    def exists(self, key: str) -> bool:
        return self.client.exists(f"cache_blob:{key}") == 1

    def set_blob(self, key: str, data: bytes, ttl: Optional[int] = None):
        k = f"cache_blob:{key}"
        self.client.set(k, data)
        if ttl:
            self.client.expire(k, ttl)

    def get_blob(self, key: str) -> Optional[bytes]:
        return self.client.get(f"cache_blob:{key}")

    def delete(self, key: str):
        self.client.delete(f"cache_blob:{key}")
        self.client.delete(f"cache_meta:{key}")


class DiskStore:
    """
    Disk-based storage with atomic write and simple eviction helpers.
    """
    def __init__(self, base_path: str = None):
        self.base = Path(base_path or config.DISK_CACHE_PATH)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # sanitize key to avoid path traversal
        safe = hashlib.sha256(key.encode()).hexdigest()
        # use first 2 chars as shard dir
        sub = safe[:2]
        return self.base / sub / safe

    def write(self, key: str, data: bytes) -> Tuple[Path, int]:
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_suffix(".tmp")
        with open(temp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, p)  # atomic move
        size = p.stat().st_size
        return p, size

    def read(self, key: str) -> Optional[bytes]:
        p = self._path_for(key)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def delete(self, key: str):
        p = self._path_for(key)
        if p.exists():
            p.unlink(missing_ok=True)

    def total_size(self) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(self.base):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except Exception:
                    pass
        return total

    def evict_least_recently_used(self, target_free_bytes: int):
        """
        Evict files until target_free_bytes is available.
        This is a simple LRU based on file mtime.
        """
        if target_free_bytes <= 0:
            return

        files = []
        for dirpath, _, filenames in os.walk(self.base):
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    files.append((p.stat().st_mtime, p))
                except Exception:
                    pass
        files.sort()  # oldest first (smallest mtime)
        freed = 0
        for mtime, p in files:
            try:
                size = p.stat().st_size
                p.unlink()
                freed += size
                if freed >= target_free_bytes:
                    break
            except Exception as e:
                logger.exception("Error evicting file %s: %s", p, e)
