# API routes for offline analytics
"""
FastAPI routes for ingesting offline events and reporting/exports.
- POST /offline-analytics/ingest -> accept a batch of events (from client when reconnecting)
- GET /offline-analytics/events/{id} -> read event metadata
- POST /offline-analytics/process -> trigger processing (admin)
- GET /offline-analytics/metrics -> query aggregated metrics (simple)
- POST /offline-analytics/export -> request export for a time window (admin)
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from .database import get_db, init_db
from .service import OfflineAnalyticsService
from .models import BatchIngestRequest, RawEventRead
from .tasks import process_events_task
from .export import export_aggregated_metrics, export_raw_events
from datetime import datetime
import logging

router = APIRouter(prefix="/offline-analytics", tags=["Offline Analytics"])
logger = logging.getLogger("offline_analytics.routes")


@router.on_event("startup")
def _startup():
    init_db()


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
def ingest_batch(body: BatchIngestRequest, db: Session = Depends(get_db)):
    """
    Endpoint used by clients to upload a batch of events collected while offline.
    Returns a summary with accepted count.
    """
    svc = OfflineAnalyticsService(db)
    inserted = svc.ingest_batch(body.events)
    return {"accepted": len(inserted)}


@router.get("/events/{event_id}", response_model=RawEventRead)
def get_event(event_id: int, db: Session = Depends(get_db)):
    evt = db.query.__self__.query  # placeholder to avoid lint error; we'll use direct query
    from .models import RawEvent
    item = db.query(RawEvent).filter(RawEvent.id == event_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Event not found")
    return item


@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
def trigger_processing(background_tasks: BackgroundTasks, batch_size: int = None):
    """
    Trigger processing via Celery task. This endpoint is safe for admins/cron.
    Returns task id.
    """
    task = process_events_task.delay(batch_size)
    return {"task_id": task.id}


@router.get("/metrics")
def query_metrics(metric_name: str = None, start: str = None, end: str = None, db: Session = Depends(get_db)):
    """
    Simple metrics query: filter aggregated_metrics table.
    Query params:
        - metric_name (optional)
        - start, end: ISO datetimes (optional)
    """
    from .models import AggregatedMetric
    q = db.query(AggregatedMetric)
    if metric_name:
        q = q.filter(AggregatedMetric.metric_name == metric_name)
    if start:
        q = q.filter(AggregatedMetric.bucket_start >= datetime.fromisoformat(start))
    if end:
        q = q.filter(AggregatedMetric.bucket_start < datetime.fromisoformat(end))
    rows = q.order_by(AggregatedMetric.bucket_start.asc()).all()
    return [{"metric_name": r.metric_name, "bucket_start": r.bucket_start.isoformat(), "value": r.value, "metadata": r.metadata} for r in rows]


@router.post("/export", status_code=status.HTTP_202_ACCEPTED)
def request_export(start: str, end: str, background_tasks: BackgroundTasks):
    """
    Schedule an export of aggregated metrics and raw events between start and end (ISO datetimes).
    For simplicity we run export synchronously in background task — in prod, trigger Celery worker.
    """
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    # Use background tasks to avoid blocking
    background_tasks.add_task(export_aggregated_metrics, s, e)
    background_tasks.add_task(export_raw_events, s, e)
    return {"status": "export_scheduled", "start": s.isoformat(), "end": e.isoformat()}
