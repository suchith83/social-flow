# Export processed analytics data
"""
Export utilities: create exports of aggregated metrics and raw events.
Supports CSV export and JSON dump for downstream ETL.
"""

import os
import csv
import json
from datetime import datetime
from .config import get_config
from .database import SessionLocal
from .models import AggregatedMetric, RawEvent
from sqlalchemy.orm import Session
import logging

config = get_config()
logger = logging.getLogger("offline_analytics.export")


def export_aggregated_metrics(start_dt, end_dt, out_path=None):
    out_path = out_path or config.EXPORT_PATH
    os.makedirs(out_path, exist_ok=True)
    fname = f"aggregated_metrics_{start_dt.isoformat()}_{end_dt.isoformat()}.csv"
    fpath = os.path.join(out_path, fname)
    db: Session = SessionLocal()
    try:
        rows = db.query(AggregatedMetric).filter(AggregatedMetric.bucket_start >= start_dt, AggregatedMetric.bucket_start < end_dt).all()
        with open(fpath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["metric_name", "bucket_start", "bucket_end", "value", "metadata"])
            for r in rows:
                writer.writerow([r.metric_name, r.bucket_start.isoformat(), r.bucket_end.isoformat(), r.value, json.dumps(r.metadata or {})])
        return fpath
    finally:
        db.close()


def export_raw_events(start_dt, end_dt, out_path=None, batch_size=1000):
    out_path = out_path or config.EXPORT_PATH
    os.makedirs(out_path, exist_ok=True)
    fname = f"raw_events_{start_dt.isoformat()}_{end_dt.isoformat()}.ndjson"
    fpath = os.path.join(out_path, fname)
    db: Session = SessionLocal()
    try:
        q = db.query(RawEvent).filter(RawEvent.ingested_at >= start_dt, RawEvent.ingested_at < end_dt).yield_per(batch_size)
        with open(fpath, "w", encoding="utf-8") as fh:
            for r in q:
                fh.write(json.dumps({
                    "id": r.id,
                    "device_id": r.device_id,
                    "user_id": r.user_id,
                    "event_type": r.event_type,
                    "event_payload": r.event_payload,
                    "occurred_at": r.occurred_at.isoformat(),
                    "ingested_at": r.ingested_at.isoformat() if r.ingested_at else None,
                    "status": r.status
                }, default=str) + "\n")
        return fpath
    finally:
        db.close()
