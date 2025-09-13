# Data retention and cleanup logic
"""
Retention and privacy enforcement utilities: remove old raw events and aggregated data
according to DATA_RETENTION_DAYS, and optionally mask or delete PII.
"""

from .config import get_config
from .database import SessionLocal
from .models import RawEvent, AggregatedMetric
from datetime import datetime, timedelta, timezone
import logging

config = get_config()
logger = logging.getLogger("offline_analytics.retention")


def run_retention_cleanup():
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=config.DATA_RETENTION_DAYS)
        # delete raw events older than cutoff
        deleted_raw = db.query(RawEvent).filter(RawEvent.ingested_at < cutoff).delete(synchronize_session=False)
        # Optionally delete aggregated metrics older than cutoff (choose policy)
        deleted_agg = db.query(AggregatedMetric).filter(AggregatedMetric.bucket_start < cutoff).delete(synchronize_session=False)
        db.commit()
        logger.info("Retention cleanup removed raw=%s agg=%s rows older than %s", deleted_raw, deleted_agg, cutoff.isoformat())
        return {"deleted_raw": deleted_raw, "deleted_agg": deleted_agg}
    finally:
        db.close()
