"""
Pydantic schemas for live streaming API validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class LiveStreamCreate(BaseModel):
    """Schema for creating a live stream."""
    title: str = Field(..., min_length=1, max_length=200, description="Stream title")
    description: Optional[str] = Field(None, max_length=2000, description="Stream description")
    tags: Optional[List[str]] = Field(None, max_length=10, description="Stream tags")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")

    @validator('title')
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

    @validator('tags')
    def validate_tags(cls, v):
        if v:
            # Validate each tag
            for tag in v:
                if not tag.strip():
                    raise ValueError('Tags cannot be empty')
                if len(tag) > 50:
                    raise ValueError('Each tag must be less than 50 characters')
            # Remove duplicates and strip
            v = list(set(tag.strip() for tag in v))
        return v


class LiveStreamUpdate(BaseModel):
    """Schema for updating a live stream."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = Field(None, max_length=10)
    thumbnail_url: Optional[str] = Field(None)

    @validator('title')
    def title_not_empty(cls, v):
        if v is not None and not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip() if v else v

    @validator('tags')
    def validate_tags(cls, v):
        if v:
            # Validate each tag
            for tag in v:
                if not tag.strip():
                    raise ValueError('Tags cannot be empty')
                if len(tag) > 50:
                    raise ValueError('Each tag must be less than 50 characters')
            # Remove duplicates and strip
            v = list(set(tag.strip() for tag in v))
        return v


class LiveStreamResponse(BaseModel):
    """Schema for live stream response."""
    stream_id: str
    user_id: str
    title: str
    description: Optional[str]
    tags: Optional[List[str]]
    thumbnail_url: Optional[str]
    channel_arn: Optional[str]
    ingest_endpoint: Optional[str]
    stream_key: str
    playback_url: str
    status: str
    viewer_count: int
    is_private: bool
    chat_enabled: bool
    recording_enabled: bool
    started_at: Optional[str]
    ended_at: Optional[str]
    created_at: str
    updated_at: str


class LiveStreamListResponse(BaseModel):
    """Schema for live streams list response."""
    streams: List[LiveStreamResponse]
    total: int
    page: int
    limit: int


class StreamViewerResponse(BaseModel):
    """Schema for stream viewer response."""
    viewer_id: str
    joined_at: str
    watch_duration: Optional[int]
    is_active: bool


class StreamAnalyticsResponse(BaseModel):
    """Schema for stream analytics response."""
    stream_id: str
    time_range: str
    viewer_count: int
    peak_viewers: int
    total_views: int
    average_watch_time: float
    engagement_rate: float
    generated_at: str