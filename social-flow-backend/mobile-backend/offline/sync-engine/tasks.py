# Async tasks for sync jobs
"""
Celery tasks for background sync maintenance: tombstone cleanup and periodic compaction.
"""

from celery import Celery
from celery.utils.log import get_task_logger
from .config import get_config
from .database import SessionLocal, init_db
from .service import SyncService
from datetime import datetime, timedelta, timezone

config = get_config()
celery_app = Celery("sync_engine_tasks", broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)
logger = get_task_logger(__name__)

# ensure DB ready
init_db()


@celery_app.task(bind=True)
def tombstone_cleanup_task(self, older_than_seconds: int = None):
    db = SessionLocal()
    try:
        older_secs = older_than_seconds or config.TOMBSTONE_RETENTION_SECONDS
        cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=older_secs)
        svc = SyncService(db)
        res = svc.compact_tombstones(cutoff)
        logger.info("Tombstone cleanup result: %s", res)
        return res
    finally:
        db.close()
