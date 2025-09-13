# Service layer: caching logic, TTL management
"""
High-level service orchestrating cache reads/writes, prefetching, eviction, and background tasks.
"""

import os
import logging
from typing import Optional
from .config import get_config
from .storage import RedisStore, DiskStore, StorageError
from .repository import CacheRepository
from .database import get_db
from .models import CacheOrigin
from .utils import sha256, normalize_key, stream_fetch
from sqlalchemy.orm import Session

logger = logging.getLogger("content_caching.service")
config = get_config()


class ContentCacheService:
    """
    Public API used by routes and background tasks.
    """

    def __init__(self, db: Session):
        self.db = db
        # init stores lazily; Redis may not be available in test env
        try:
            self.redis = RedisStore(config.REDIS_URL)
        except Exception:
            self.redis = None
            logger.info("Redis not available; using disk-only")
        self.disk = DiskStore(config.DISK_CACHE_PATH)

    def has(self, key: str) -> bool:
        key = normalize_key(key)
        # check redis first
        if self.redis:
            try:
                if self.redis.exists(key):
                    CacheRepository.mark_accessed(self.db, key)
                    return True
            except Exception:
                logger.exception("Redis check failed; falling back to disk")
        # check disk
        ok = self.disk.exists(key)
        if ok:
            CacheRepository.mark_accessed(self.db, key)
        return ok

    def get(self, key: str) -> Optional[bytes]:
        key = normalize_key(key)
        # Redis blob
        if self.redis:
            try:
                data = self.redis.get_blob(key)
                if data:
                    CacheRepository.mark_accessed(self.db, key)
                    return data
            except Exception:
                logger.exception("Redis get failed; falling back to disk")
        data = self.disk.read(key)
        if data:
            CacheRepository.mark_accessed(self.db, key)
        return data

    def put_from_bytes(self, key: str, data: bytes, origin: CacheOrigin = CacheOrigin.MANUAL, ttl: Optional[int] = None, metadata: dict = None):
        key = normalize_key(key)
        ttl = ttl or config.DEFAULT_TTL
        ch = sha256(data)
        # attempt to write to disk (master copy), then optionally populate redis
        try:
            p, size = self.disk.write(key, data)
        except Exception as e:
            logger.exception("Disk write failed")
            raise

        # store small blob in redis if available and small (<1MB)
        if self.redis and size <= 1024 * 1024:
            try:
                self.redis.set_blob(key, data, ttl=ttl)
                meta = {"path": str(p), "size": size}
                self.redis.set_meta(key, {"checksum": ch, "size": size, "origin": origin.value, "metadata": metadata or {}}, ttl=ttl)
            except Exception:
                logger.exception("Redis write failed; continuing with disk copy")

        # update DB metadata
        CacheRepository.create_or_update(self.db, key=key, size_bytes=size, checksum=ch, origin=origin, ttl=ttl, metadata=metadata or {})
        self._ensure_disk_under_limit()
        return {"key": key, "size": size, "checksum": ch}

    def put_from_url(self, key: str, url: str, origin: CacheOrigin = CacheOrigin.REMOTE, ttl: Optional[int] = None, metadata: dict = None):
        """
        Stream remote content and store. This function streams content to disk to minimize memory usage.
        """
        key = normalize_key(key)
        ttl = ttl or config.DEFAULT_TTL
        hasher = hashlib.sha256()
        temp_chunks = []
        total = 0
        # stream fetch and write to disk incrementally
        # We'll write to a temporary file via DiskStore.write (which expects full bytes). To avoid large memory
        # we write chunks to a temporary file path directly here.
        temp_file_path = None
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(delete=False) as tf:
            temp_file_path = tf.name
            for chunk in stream_fetch(url):
                tf.write(chunk)
                hasher.update(chunk)
                total += len(chunk)
        # Read bytes back in streaming-friendly way for DiskStore.write
        with open(temp_file_path, "rb") as f:
            data = f.read()
        os.unlink(temp_file_path)
        return self.put_from_bytes(key, data, origin=origin, ttl=ttl, metadata=metadata)

    def delete(self, key: str):
        key = normalize_key(key)
        if self.redis:
            try:
                self.redis.delete(key)
            except Exception:
                logger.exception("Redis delete failed")
        self.disk.delete(key)
        item = CacheRepository.get(self.db, key)
        if item:
            self.db.delete(item)
            self.db.commit()
        return True

    def _ensure_disk_under_limit(self):
        total = self.disk.total_size()
        if total > config.MAX_DISK_USAGE_BYTES:
            # compute bytes to free
            to_free = total - config.MAX_DISK_USAGE_BYTES + config.MIN_FREE_DISK_BYTES
            logger.info("Disk usage %s bytes > limit %s. Evicting %s bytes", total, config.MAX_DISK_USAGE_BYTES, to_free)
            # Evict metadata first
            freed_md = CacheRepository.evict_oldest(self.db, to_free)
            # Evict physical files via disk store if still needed
            remain = to_free - freed_md
            if remain > 0:
                self.disk.evict_least_recently_used(remain)

    def prefetch_keys(self, keys: list, origin_map: dict = None):
        """
        Enqueue prefetch jobs (used by routes and tasks). For simplicity, tasks will call put_from_url
        with provided url in metadata or origin_map.
        """
        # This method should be lightweight: it returns list of jobs scheduled
        from .tasks import prefetch_task
        jobs = []
        for key in keys:
            info = origin_map.get(key) if origin_map else None
            url = None
            if info and isinstance(info, dict):
                url = info.get("url")
            if url:
                job = prefetch_task.delay(key, url)
                jobs.append(job.id)
        return jobs
