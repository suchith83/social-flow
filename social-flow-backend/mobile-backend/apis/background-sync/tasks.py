# Async job queue integration (Celery/RQ/custom)
"""
Celery tasks for background sync.
Handles retries, batching, and error handling.
"""

from celery import Celery
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from .config import get_config
from .repository import SyncJobRepository
from .models import SyncStatus

config = get_config()

celery_app = Celery(
    "background_sync",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
)

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@celery_app.task(bind=True, max_retries=config.MAX_RETRIES)
def run_background_sync(self, job_id: int):
    """
    Background sync task.
    Handles retries with exponential backoff if failures occur.
    """
    db = SessionLocal()
    try:
        SyncJobRepository.update_status(db, job_id, SyncStatus.IN_PROGRESS)

        # --- Simulated background sync work ---
        import time, random
        time.sleep(2)  # mimic API call / batch process
        if random.random() < 0.2:
            raise Exception("Random simulated failure")

        result = {"message": f"Job {job_id} sync completed successfully"}
        SyncJobRepository.update_status(db, job_id, SyncStatus.SUCCESS, result)
        return result

    except Exception as exc:
        countdown = config.RETRY_BACKOFF * (2 ** self.request.retries)
        try:
            self.retry(exc=exc, countdown=countdown)
        except self.MaxRetriesExceededError:
            SyncJobRepository.update_status(db, job_id, SyncStatus.FAILED, {"error": str(exc)})
            raise
    finally:
        db.close()
