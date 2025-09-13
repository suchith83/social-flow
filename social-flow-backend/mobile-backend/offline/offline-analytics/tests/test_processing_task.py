# Unit tests for batch processing tasks
"""
Unit test for the processing service (invokes process_pending_batch).
"""

import pytest
from mobile_backend.offline.offline_analytics.database import init_db, SessionLocal
from mobile_backend.offline.offline_analytics.repository import RawEventRepository, AggregationRepository
from mobile_backend.offline.offline_analytics.service import OfflineAnalyticsService
from datetime import datetime, timezone

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield

def test_process_single_event():
    db = SessionLocal()
    # insert raw event
    now = datetime.now(timezone.utc)
    evt = RawEventRepository.insert_event(db, device_id="d1", user_id="u1", event_type="open", payload={"x": 1}, occurred_at=now, size_bytes=100)
    svc = OfflineAnalyticsService(db)
    processed = svc.process_pending_batch(batch_size=10)
    assert processed >= 1
    # ensure aggregation exists
    # fetch aggregation row
    rows = db.query.__self__  # avoid lint; use direct query instead
    from mobile_backend.offline.offline_analytics.models import AggregatedMetric
    metrics = db.query(AggregatedMetric).filter(AggregatedMetric.metric_name == "open").all()
    assert len(metrics) >= 1
    db.close()
