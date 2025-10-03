"""
Video Pydantic schemas for API request/response validation.

This module provides schemas for video-related operations including:
- Video upload and processing
- Video metadata management
- Video streaming and playback
- Video analytics
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import Field, field_validator

from app.schemas.base import BaseDBSchema, BaseSchema


# Enums
class VideoStatus(str):
    """Video processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class VideoVisibility(str):
    """Video visibility settings."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"
    FOLLOWERS_ONLY = "followers_only"


class VideoQuality(str):
    """Video quality levels."""
    QUALITY_360P = "360p"
    QUALITY_480P = "480p"
    QUALITY_720P = "720p"
    QUALITY_1080P = "1080p"
    QUALITY_4K = "4k"


# Base schemas
class VideoBase(BaseSchema):
    """Base video schema with common fields."""
    
    title: str = Field(..., min_length=1, max_length=255, description="Video title")
    description: Optional[str] = Field(None, max_length=5000, description="Video description")
    tags: list[str] = Field(default_factory=list, max_length=20, description="Video tags")
    visibility: str = Field(default="public", description="Video visibility")
    is_age_restricted: bool = Field(default=False, description="Age restriction flag")
    allow_comments: bool = Field(default=True, description="Allow comments")
    allow_likes: bool = Field(default=True, description="Allow likes")
    is_monetized: bool = Field(default=False, description="Monetization enabled")


# Create schemas
class VideoCreate(VideoBase):
    """Schema for video upload initiation."""
    
    original_filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    duration: Optional[float] = Field(None, gt=0, description="Video duration in seconds")


class VideoUploadInit(BaseSchema):
    """Initialize video upload."""
    
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., gt=0, le=10*1024*1024*1024, description="File size in bytes (max 10GB)")
    content_type: str = Field(..., pattern=r'^video/', description="MIME type")
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate video MIME type."""
        allowed = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/webm', 'video/x-msvideo']
        if v not in allowed:
            raise ValueError(f'Content type must be one of: {", ".join(allowed)}')
        return v


class VideoUploadComplete(BaseSchema):
    """Complete video upload."""
    
    video_id: UUID = Field(..., description="Video ID")
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    tags: list[str] = Field(default_factory=list)
    visibility: str = Field(default="public")


# Update schemas
class VideoUpdate(BaseSchema):
    """Schema for updating video metadata."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    tags: Optional[list[str]] = Field(None, max_length=20)
    visibility: Optional[str] = None
    thumbnail_url: Optional[str] = None
    is_age_restricted: Optional[bool] = None
    allow_comments: Optional[bool] = None
    allow_likes: Optional[bool] = None
    is_monetized: Optional[bool] = None


# Response schemas
class VideoResponse(BaseDBSchema):
    """Basic video response."""
    
    # Identity
    owner_id: UUID
    title: str
    description: Optional[str] = None
    
    # Media files
    original_file_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    hls_master_url: Optional[str] = None
    dash_manifest_url: Optional[str] = None
    
    # Metadata
    duration: Optional[float] = None
    file_size: Optional[int] = None
    status: str
    visibility: str
    tags: list[str] = []
    
    # Settings
    age_restricted: bool = False
    
    # Stats
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    
    # Timestamps
    uploaded_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


class VideoDetailResponse(VideoResponse):
    """Detailed video response with processing info."""
    
    # Processing details
    processing_progress: int = 0
    processing_error: Optional[str] = None
    mediaconvert_job_id: Optional[str] = None
    
    # Available qualities
    available_qualities: list[str] = []
    
    # Analytics
    average_watch_time: Optional[float] = None
    completion_rate: Optional[float] = None
    engagement_rate: Optional[float] = None
    
    # Monetization
    revenue_total: float = 0.0
    ad_impressions: int = 0
    
    # Geographic
    allowed_countries: Optional[list[str]] = None
    blocked_countries: list[str] = []


class VideoPublicResponse(BaseDBSchema):
    """Public video response (minimal data for listings)."""
    
    owner_id: UUID
    title: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[float] = None
    status: str
    visibility: str
    
    # Stats
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    
    # Timestamps
    published_at: Optional[datetime] = None


# Upload schemas
class VideoUploadURL(BaseSchema):
    """Pre-signed upload URL response."""
    
    upload_url: str = Field(..., description="Pre-signed S3 upload URL")
    video_id: UUID = Field(..., description="Video ID")
    expires_in: int = Field(..., description="URL expiration time in seconds")


# Streaming schemas
class VideoStreamingURLs(BaseSchema):
    """Streaming URLs for video playback."""
    
    hls_url: Optional[str] = Field(None, description="HLS master playlist URL")
    dash_url: Optional[str] = Field(None, description="DASH manifest URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    poster_url: Optional[str] = Field(None, description="Poster image URL")
    
    # Available qualities with direct URLs
    qualities: dict[str, str] = Field(default_factory=dict, description="Quality -> URL mapping")


# Analytics schemas
class VideoAnalytics(BaseSchema):
    """Video analytics data."""
    
    video_id: UUID
    views: int = 0  # Alias for views_total for backwards compatibility
    views_total: int = 0
    views_today: int = 0
    views_week: int = 0
    views_month: int = 0
    
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    
    average_watch_time: float = 0.0
    completion_rate: float = 0.0
    engagement_rate: float = 0.0
    
    # Geographic distribution
    top_countries: list[dict[str, int]] = []
    
    # Traffic sources
    traffic_sources: dict[str, int] = {}
    
    # Device types
    device_types: dict[str, int] = {}


# List and filter schemas
class VideoListFilters(BaseSchema):
    """Filters for video listing."""
    
    user_id: Optional[UUID] = None
    status: Optional[str] = None
    visibility: Optional[str] = None
    is_monetized: Optional[bool] = None
    tags: Optional[list[str]] = None
    search: Optional[str] = Field(None, description="Search by title or description")
    uploaded_after: Optional[datetime] = None
    uploaded_before: Optional[datetime] = None
    min_views: Optional[int] = Field(None, ge=0)
    min_duration: Optional[float] = Field(None, ge=0)
    max_duration: Optional[float] = Field(None, ge=0)


class VideoSortOptions(BaseSchema):
    """Video sorting options."""
    
    sort_by: str = Field(
        default="published_at",
        description="Sort field: published_at, views_count, likes_count, created_at"
    )
    sort_order: str = Field(default="desc", pattern=r'^(asc|desc)$')


# Batch operations
class VideoBatchUpdate(BaseSchema):
    """Batch update videos."""
    
    video_ids: list[UUID] = Field(..., min_length=1, max_length=100)
    visibility: Optional[str] = None
    is_monetized: Optional[bool] = None
    tags_add: Optional[list[str]] = None
    tags_remove: Optional[list[str]] = None


class VideoBatchDelete(BaseSchema):
    """Batch delete videos."""
    
    video_ids: list[UUID] = Field(..., min_length=1, max_length=100)
    permanent: bool = Field(default=False, description="Permanent deletion (cannot be undone)")
