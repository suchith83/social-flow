"""
Pydantic models for upload endpoints and internal events
"""

from pydantic import BaseModel, Field
from typing import Optional
import datetime


class InitUploadRequest(BaseModel):
    filename: str
    total_bytes: int
    mime_type: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = {}


class InitUploadResponse(BaseModel):
    upload_id: str
    chunk_size: int
    upload_url: Optional[str] = None  # for direct presigned
    expires_in: int


class ChunkCompleteEvent(BaseModel):
    upload_id: str
    chunk_index: int
    bytes_received: int
    user_id: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class UploadCompleteEvent(BaseModel):
    upload_id: str
    file_path: str
    total_bytes: int
    user_id: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
