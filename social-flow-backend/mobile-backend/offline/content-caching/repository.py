# Repository for cache reads/writes
"""
Repository to interact with DB metadata table 'cached_items'.
Keeps cached item records, helps with eviction decisions, and LRU updates.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func
from .models import CachedItem, CacheOrigin
from datetime import datetime
from typing import Optional, List


class CacheRepository:

    @staticmethod
    def create_or_update(db: Session, key: str, size_bytes: int, checksum: Optional[str], origin: CacheOrigin, ttl: int, metadata: dict):
        item = db.query(CachedItem).filter(CachedItem.key == key).first()
        if item:
            item.size_bytes = size_bytes
            if checksum:
                item.checksum = checksum
            item.origin = origin.value if hasattr(origin, "value") else origin
            item.ttl = ttl
            item.metadata = metadata or {}
            item.is_stale = False
            item.accessed_at = func.now()
        else:
            item = CachedItem(
                key=key,
                size_bytes=size_bytes,
                checksum=checksum,
                origin=origin.value if hasattr(origin, "value") else origin,
                ttl=ttl,
                metadata=metadata or {},
            )
            db.add(item)
        db.commit()
        db.refresh(item)
        return item

    @staticmethod
    def mark_accessed(db: Session, key: str):
        item = db.query(CachedItem).filter(CachedItem.key == key).first()
        if item:
            item.accessed_at = func.now()
            db.commit()
            db.refresh(item)
        return item

    @staticmethod
    def get(db: Session, key: str) -> Optional[CachedItem]:
        return db.query(CachedItem).filter(CachedItem.key == key).first()

    @staticmethod
    def list_stale(db: Session, limit: int = 100) -> List[CachedItem]:
        return db.query(CachedItem).filter(CachedItem.is_stale == True).order_by(CachedItem.accessed_at.asc()).limit(limit).all()

    @staticmethod
    def evict_oldest(db: Session, bytes_to_free: int) -> int:
        """
        Evict oldest items until bytes_to_free are released, returning bytes freed.
        Note: this only removes metadata records; physical file deletion must be handled by service.
        """
        freed = 0
        items = db.query(CachedItem).order_by(CachedItem.accessed_at.asc()).all()
        for item in items:
            freed += item.size_bytes
            db.delete(item)
            db.commit()
            if freed >= bytes_to_free:
                break
        return freed
