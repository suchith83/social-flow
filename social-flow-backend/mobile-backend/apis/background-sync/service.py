# Orchestration, scheduling, retries
"""
Service layer for orchestrating background sync logic.
This interacts with repository + Celery tasks.
"""

from sqlalchemy.orm import Session
from .repository import SyncJobRepository
from .models import SyncJobCreate, SyncStatus
from .tasks import run_background_sync


class SyncService:

    @staticmethod
    def create_and_dispatch_job(db: Session, job_data: SyncJobCreate):
        job = SyncJobRepository.create_job(db, job_data)
        # Trigger Celery background task
        run_background_sync.delay(job.id)
        return job

    @staticmethod
    def get_job_status(db: Session, job_id: int):
        return SyncJobRepository.get_job(db, job_id)

    @staticmethod
    def list_jobs(db: Session, limit: int = 50, offset: int = 0):
        return SyncJobRepository.list_jobs(db, limit, offset)
