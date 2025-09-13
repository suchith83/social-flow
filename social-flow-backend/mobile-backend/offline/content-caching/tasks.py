# Async background tasks for cache refresh/invalidation
"""
Celery tasks for background prefetching and maintenance.
- prefetch_task: fetch a URL and cache it
- periodic_maintenance: evict and mark stale items
"""

from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger
from .config import get_config
from .storage import DiskStore, RedisStore
from .database import engine, SessionLocal, init_db
from .repository import CacheRepository
from .models import CacheOrigin
from .service import ContentCacheService
import os

config = get_config()
celery_app = Celery("content_cache_tasks", broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)
logger = get_task_logger(__name__)

# Ensure DB exists before tasks run
init_db()


@celery_app.task(bind=True, max_retries=3, acks_late=True)
def prefetch_task(self, key: str, url: str):
    """
    Fetch remote content and store in cache. Retries on failure.
    """
    db = SessionLocal()
    try:
        svc = ContentCacheService(db)
        svc.put_from_url(key, url, origin=CacheOrigin.PREFETCH)
        return {"key": key, "status": "cached"}
    except Exception as exc:
        logger.exception("Prefetch failed for %s", key)
        try:
            self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        except self.MaxRetriesExceededError:
            logger.exception("Prefetch failed permanently for %s", key)
            raise
    finally:
        db.close()


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Example: run maintenance every hour to mark stale items & evict if needed
    sender.add_periodic_task(3600.0, periodic_maintenance.s(), name="cache-maintenance-every-hour")


@celery_app.task
def periodic_maintenance():
    """
    Mark stale items older than STALE_THRESHOLD and evict if disk usage exceeded.
    """
    db = SessionLocal()
    try:
        from datetime import datetime, timedelta
        threshold = datetime.utcnow().timestamp() - config.STALE_THRESHOLD
        # Mark items as stale (simplified logic)
        # In SQLAlchemy we'd compare timestamps properly; keep simple for example
        items = CacheRepository.list_stale(db, limit=500)
        # here we could revalidate with origin or refresh; for now, mark and optionally evict
        disk = DiskStore()
        if disk.total_size() > config.MAX_DISK_USAGE_BYTES:
            to_free = disk.total_size() - config.MAX_DISK_USAGE_BYTES + config.MIN_FREE_DISK_BYTES
            CacheRepository.evict_oldest(db, to_free)
            disk.evict_least_recently_used(to_free)
        return {"status": "maintenance-completed"}
    finally:
        db.close()
