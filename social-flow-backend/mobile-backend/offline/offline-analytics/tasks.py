# Async tasks for batch processing
"""
Celery tasks for background processing (consume raw events, aggregate, export).
"""

from celery import Celery
from celery.utils.log import get_task_logger
from .config import get_config
from .database import SessionLocal, init_db
from .service import OfflineAnalyticsService
import time
import logging

config = get_config()
celery_app = Celery("offline_analytics", broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)
logger = get_task_logger(__name__)

# Ensure DB exists
init_db()


@celery_app.task(bind=True, max_retries=3, acks_late=True)
def process_events_task(self, batch_size: int = None):
    """
    Task: process a batch of pending events into aggregated metrics.
    """
    batch_size = batch_size or config.PROCESS_BATCH_SIZE
    db = SessionLocal()
    try:
        svc = OfflineAnalyticsService(db)
        processed = svc.process_pending_batch(batch_size)
        logger.info("Processed %s events", processed)
        return {"processed": processed}
    except Exception as exc:
        logger.exception("Processing failed")
        try:
            countdown = 60 * (2 ** self.request.retries)
            self.retry(exc=exc, countdown=countdown)
        except self.MaxRetriesExceededError:
            logger.exception("Processing failed permanently")
            raise
    finally:
        db.close()


@celery_app.task(bind=True)
def periodic_aggregation(self):
    """
    Wrap process_events_task for periodic runs. This function can be hooked into Celery beat.
    """
    return process_events_task.delay()
