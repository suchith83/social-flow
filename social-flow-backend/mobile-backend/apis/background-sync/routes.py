# API endpoints for sync
"""
API routes for Background Sync.
Exposes endpoints for creating jobs, checking status, and listing jobs.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .service import SyncService
from .models import SyncJobCreate, SyncJobRead, SyncJobListResponse
from .database import get_db

router = APIRouter(prefix="/background-sync", tags=["Background Sync"])


@router.post("/", response_model=SyncJobRead)
def create_sync_job(job: SyncJobCreate, db: Session = Depends(get_db)):
    return SyncService.create_and_dispatch_job(db, job)


@router.get("/{job_id}", response_model=SyncJobRead)
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = SyncService.get_job_status(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/", response_model=SyncJobListResponse)
def list_jobs(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    jobs = SyncService.list_jobs(db, limit, offset)
    return {"jobs": jobs}
