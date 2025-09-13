# FastAPI routes for sync operations
"""
FastAPI routes for the sync engine.
Endpoints:
- POST /sync/push -> client pushes a batch of operations
- POST /sync/pull -> client pulls changes since a seq
- POST /sync/resolve -> optional endpoint to resolve conflicts (admin/custom)
- GET /sync/last_seq -> get latest global sequence
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .database import get_db, init_db
from .service import SyncService
from .models import PushBatchRequest, PullRequest
from .config import get_config
import logging

router = APIRouter(prefix="/sync", tags=["Offline Sync"])
logger = logging.getLogger("sync.routes")

@router.on_event("startup")
def _startup():
    init_db()


@router.post("/push", status_code=status.HTTP_200_OK)
def push_changes(body: PushBatchRequest, db: Session = Depends(get_db)):
    svc = SyncService(db)
    if len(body.ops) > get_config().MAX_PUSH_BATCH_SIZE:
        raise HTTPException(status_code=400, detail="Batch too large")
    res = svc.apply_push_batch(body)
    return res


@router.post("/pull", status_code=status.HTTP_200_OK)
def pull_changes(body: PullRequest, db: Session = Depends(get_db)):
    svc = SyncService(db)
    res = svc.pull_changes(body)
    return res


@router.get("/last_seq")
def last_seq(db: Session = Depends(get_db)):
    from .repository import SyncRepository
    last = SyncRepository.get_last_seq(db)
    return {"last_seq": last}
