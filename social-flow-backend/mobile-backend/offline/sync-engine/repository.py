# Repository layer for sync job data
"""
Repository layer for accessing sync_items and change_log with transactional helpers.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from .models import SyncItem, ChangeLog, ChangeType
from typing import Optional, List, Dict, Any
from datetime import datetime


class SyncRepository:
    @staticmethod
    def get_item(db: Session, key: str) -> Optional[SyncItem]:
        return db.query(SyncItem).filter(SyncItem.key == key).first()

    @staticmethod
    def upsert_item(db: Session, key: str, data: dict, server_version: int, client_version: int, deleted: bool = False) -> SyncItem:
        item = db.query(SyncItem).filter(SyncItem.key == key).first()
        if item:
            item.data = data
            item.server_version = server_version
            item.client_version = client_version
            item.deleted = deleted
        else:
            item = SyncItem(key=key, data=data, server_version=server_version, client_version=client_version, deleted=deleted)
            db.add(item)
        db.commit()
        db.refresh(item)
        return item

    @staticmethod
    def mark_deleted(db: Session, key: str, server_version: int):
        item = db.query(SyncItem).filter(SyncItem.key == key).first()
        if item:
            item.deleted = True
            item.server_version = server_version
            db.commit()
            db.refresh(item)
            return item
        # create tombstone if not present
        item = SyncItem(key=key, data=None, server_version=server_version, client_version=0, deleted=True)
        db.add(item)
        db.commit()
        db.refresh(item)
        return item

    @staticmethod
    def insert_changelog(db: Session, key: str, change_type: ChangeType, payload: dict, server_version: int) -> ChangeLog:
        # compute next seq as max(seq)+1
        last = db.query(func.max(ChangeLog.seq)).scalar()
        next_seq = (last or 0) + 1
        log = ChangeLog(seq=next_seq, key=key, change_type=change_type.value, payload=payload, server_version=server_version)
        db.add(log)
        db.commit()
        db.refresh(log)
        return log

    @staticmethod
    def get_changes_since(db: Session, since_seq: int, limit: int = 500) -> List[ChangeLog]:
        return db.query(ChangeLog).filter(ChangeLog.seq > since_seq).order_by(ChangeLog.seq.asc()).limit(limit).all()

    @staticmethod
    def get_last_seq(db: Session) -> int:
        last = db.query(func.max(ChangeLog.seq)).scalar()
        return int(last or 0)

    @staticmethod
    def delete_old_tombstones(db: Session, older_than_timestamp):
        # remove tombstones older than timestamp
        q = db.query(SyncItem).filter(SyncItem.deleted == True, SyncItem.updated_at < older_than_timestamp)
        count = q.delete(synchronize_session=False)
        db.commit()
        return count
