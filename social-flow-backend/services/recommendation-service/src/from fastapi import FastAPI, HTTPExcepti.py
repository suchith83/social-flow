from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import sys
import time

# make package importable when run from repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Lightweight logger fallback for isolated runs/tests
try:
    from common.libraries.python.monitoring.logger import get_logger
except Exception:
    import logging

    def get_logger(name: str):
        l = logging.getLogger(name)
        if not l.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            l.addHandler(h)
        l.setLevel(logging.INFO)
        return l

from src.store import InMemoryMetricsStore

logger = get_logger("analytics-service")
app = FastAPI(title="Analytics Service", version="0.1.0")

# single global store instance for this service module (sufficient for tests/local)
store = InMemoryMetricsStore()


class MetricPoint(BaseModel):
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Numeric value")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    timestamp: Optional[float] = None  # epoch seconds


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(point: MetricPoint):
    try:
        ts = point.timestamp if point.timestamp is not None else time.time()
        store.add_point(point.name, point.value, point.tags, ts)
        logger.info("ingested metric", extra={"name": point.name, "value": point.value})
        return {"status": "ingested", "name": point.name, "timestamp": ts}
    except Exception as exc:
        logger.exception("Failed to ingest metric")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def get_metrics(name: str = Query(..., description="Metric name"),
                start_ts: Optional[float] = Query(None),
                end_ts: Optional[float] = Query(None),
                limit: int = Query(100, ge=1, le=1000)):
    try:
        points = store.query(name, start_ts, end_ts)
        # apply simple limiting (most recent first)
        points = sorted(points, key=lambda p: p["timestamp"], reverse=True)[:limit]
        return {"name": name, "count": len(points), "points": points}
    except Exception as exc:
        logger.exception("Failed to query metrics")
        raise HTTPException(status_code=500, detail=str(exc))
