# Repository layer for data retrieval and storage
"""
Repository layer for raw events and aggregation.
Encapsulates DB queries for safety and testability.
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from .models import RawEvent, AggregatedMetric, IngestStatus
from datetime import datetime
from sqlalchemy import func, and_, asc
from .models import RawEventCreate as RawEventCreateSchema  # not SQLAlchemy model; used for typing


class RawEventRepository:

    @staticmethod
    def insert_event(db: Session, device_id: Optional[str], user_id: Optional[str], event_type: str, payload: dict, occurred_at: datetime, size_bytes: int) -> RawEvent:
        evt = RawEvent(
            device_id=device_id,
            user_id=user_id,
            event_type=event_type,
            event_payload=payload,
            occurred_at=occurred_at,
            size_bytes=size_bytes,
            status=IngestStatus.PENDING.value,
        )
        db.add(evt)
        db.commit()
        db.refresh(evt)
        return evt

    @staticmethod
    def bulk_insert_events(db: Session, events: List[dict]) -> List[RawEvent]:
        objs = []
        for e in events:
            objs.append(RawEvent(
                device_id=e.get("device_id"),
                user_id=e.get("user_id"),
                event_type=e["event_type"],
                event_payload=e.get("event_payload", {}),
                occurred_at=e["occurred_at"],
                size_bytes=e.get("size_bytes", 0),
                status=IngestStatus.PENDING.value,
            ))
        db.bulk_save_objects(objs)
        db.commit()
        # Note: bulk_save_objects won't populate PKs by default; fetch inserted rows if needed
        return objs

    @staticmethod
    def fetch_pending_batch(db: Session, limit: int):
        return db.query(RawEvent).filter(RawEvent.status == IngestStatus.PENDING.value).order_by(asc(RawEvent.ingested_at)).limit(limit).all()

    @staticmethod
    def update_status(db: Session, event_id: int, status: IngestStatus):
        evt = db.query(RawEvent).filter(RawEvent.id == event_id).first()
        if evt:
            evt.status = status.value
            if status == IngestStatus.PROCESSED:
                evt.processed_at = func.now()
            db.commit()
            db.refresh(evt)
        return evt


class AggregationRepository:
    @staticmethod
    def upsert_metric(db: Session, metric_name: str, bucket_start, bucket_end, increment: int = 1, metadata: dict = None):
        row = db.query(AggregatedMetric).filter(
            AggregatedMetric.metric_name == metric_name,
            AggregatedMetric.bucket_start == bucket_start
        ).first()
        if row:
            row.value = row.value + increment
            if metadata:
                row.metadata = {**(row.metadata or {}), **metadata}
        else:
            row = AggregatedMetric(metric_name=metric_name, bucket_start=bucket_start, bucket_end=bucket_end, value=increment, metadata=metadata or {})
            db.add(row)
        db.commit()
        db.refresh(row)
        return row
