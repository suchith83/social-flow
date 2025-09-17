"""
Pydantic models for thumbnail operations
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict
import datetime


class ThumbnailSpec(BaseModel):
    width: int
    height: int
    format: str = "jpeg"  # jpeg or webp
    quality: int = 85


class ThumbnailResult(BaseModel):
    video_id: str
    thumbnail_id: str
    size: str  # e.g., "320x180"
    format: str
    url: str
    phash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class ThumbnailsBatchResult(BaseModel):
    video_id: str
    thumbnails: List[ThumbnailResult]
    sprite_url: Optional[str] = None
    metadata: Optional[Dict] = {}
