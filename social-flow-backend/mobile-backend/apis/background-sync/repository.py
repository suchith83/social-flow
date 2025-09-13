# Database access for sync jobs
"""
Repository layer for managing sync jobs in DB.
Encapsulates persistence logic from business logic.
"""

from sqlalchemy.orm import Session
from .models import SyncJob, SyncStatus, SyncJobCreate


class SyncJobRepository:

    @staticmethod
    def create_job(db: Session, job_data: SyncJobCreate) -> SyncJob:
        job = SyncJob(job_name=job_data.job_name, payload=job_data.payload)
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def update_status(db: Session, job_id: int, status: SyncStatus, result: dict = None) -> SyncJob:
        job = db.query(SyncJob).filter(SyncJob.id == job_id).first()
        if job:
            job.status = status
            if result:
                job.result = result
            db.commit()
            db.refresh(job)
        return job

    @staticmethod
    def get_job(db: Session, job_id: int) -> SyncJob:
        return db.query(SyncJob).filter(SyncJob.id == job_id).first()

    @staticmethod
    def list_jobs(db: Session, limit: int = 50, offset: int = 0):
        return db.query(SyncJob).order_by(SyncJob.created_at.desc()).offset(offset).limit(limit).all()
