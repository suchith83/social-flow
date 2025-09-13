# FastAPI routes for cache API
"""
FastAPI routes for offline content caching.
Endpoints:
- GET /cache/{key} -> retrieve cached content (streamed)
- POST /cache -> store content from bytes
- POST /cache/prefetch -> schedule prefetch tasks for keys/urls
- DELETE /cache/{key} -> remove cached item
- HEAD /cache/{key} -> check presence & metadata
"""

from fastapi import APIRouter, Depends, HTTPException, Response, StreamingResponse, UploadFile, File, status
from sqlalchemy.orm import Session
from .database import get_db, init_db
from .service import ContentCacheService
from .models import CachedItemCreate, CachedItemRead
from .models import CacheOrigin
import io
import logging

logger = logging.getLogger("content_caching.routes")
router = APIRouter(prefix="/offline/cache", tags=["Offline Content Cache"])


@router.on_event("startup")
def _startup():
    init_db()


@router.head("/{key}")
def head_cache(key: str, db: Session = Depends(get_db)):
    svc = ContentCacheService(db)
    exists = svc.has(key)
    if not exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    # return minimal metadata
    item = svc.get(key)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return Response(status_code=status.HTTP_200_OK)


@router.get("/{key}")
def get_cache(key: str, db: Session = Depends(get_db)):
    svc = ContentCacheService(db)
    data = svc.get(key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cache miss")
    # Stream back as bytes. Use StreamingResponse if large.
    return Response(content=data, media_type="application/octet-stream")


@router.post("/", status_code=status.HTTP_201_CREATED)
async def put_cache(key: str, file: UploadFile = File(...), origin: str = "manual", db: Session = Depends(get_db)):
    """
    Upload binary data to cache.
    """
    svc = ContentCacheService(db)
    content = await file.read()
    res = svc.put_from_bytes(key, content, origin=CacheOrigin.MANUAL)
    return res


@router.post("/prefetch", status_code=status.HTTP_202_ACCEPTED)
def prefetch(keys: dict, db: Session = Depends(get_db)):
    """
    Request prefetch for a dict mapping key -> {url: "<url>"}.
    Example body:
    {
      "content:123": {"url": "https://example.com/content/123"},
      "content:124": {"url": "https://example.com/content/124"}
    }
    """
    svc = ContentCacheService(db)
    job_ids = svc.prefetch_keys(list(keys.keys()), origin_map=keys)
    return {"scheduled_jobs": job_ids}


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cache(key: str, db: Session = Depends(get_db)):
    svc = ContentCacheService(db)
    ok = svc.delete(key)
    if not ok:
        raise HTTPException(status_code=404, detail="Cache key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
