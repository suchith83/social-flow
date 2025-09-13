# FastAPI routes for mobile-optimized endpoints
"""
API routes for Mobile Optimized endpoints.
Supports lightweight pagination and delta sync.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from .service import MobileOptimizedService
from .models import OptimizedItemListResponse, DeltaSyncRequest, DeltaSyncResponse
from .database import get_db
from .config import get_config

config = get_config()

router = APIRouter(prefix="/mobile-optimized", tags=["Mobile Optimized"])


@router.get("/items", response_model=OptimizedItemListResponse)
def list_items(
    limit: int = Query(config.DEFAULT_PAGE_SIZE, le=config.MAX_PAGE_SIZE),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    return MobileOptimizedService.list_items(db, limit, offset)


@router.post("/delta-sync", response_model=DeltaSyncResponse)
def delta_sync(sync_request: DeltaSyncRequest, db: Session = Depends(get_db)):
    return MobileOptimizedService.delta_sync(db, sync_request)
