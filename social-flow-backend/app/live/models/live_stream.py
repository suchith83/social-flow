"""
Live Stream model and related schemas.

This module defines the LiveStream model and related Pydantic schemas
for live streaming in the Social Flow backend.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, Enum as SQLEnum, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class LiveStreamStatus(str, Enum):
    """Live stream status."""
    STARTING = "starting"
    ACTIVE = "active"
    ENDED = "ended"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Alias for compatibility
StreamStatus = LiveStreamStatus


class LiveStreamVisibility(str, Enum):
    """Live stream visibility level."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


class LiveStream(Base):
    """Live stream model for storing live streaming information."""
    
    __tablename__ = "live_streams"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)  # JSON string of tags
    category = Column(String(100), nullable=True)  # Stream category
    
    # Stream information
    stream_key = Column(String(255), nullable=False, unique=True)
    channel_arn = Column(String(500), nullable=True)  # AWS IVS channel ARN
    ingest_endpoint = Column(String(500), nullable=True)  # RTMP ingest endpoint
    playback_url = Column(String(500), nullable=True)  # HLS playback URL
    
    # Stream settings
    is_private = Column(Boolean, default=False, nullable=False)
    record_stream = Column(Boolean, default=True, nullable=False)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # Using user_id instead of owner_id
    
    # Thumbnail and preview
    thumbnail_url = Column(String(500), nullable=True)
    preview_url = Column(String(500), nullable=True)
    
    # Status and visibility
    status = Column(SQLEnum(LiveStreamStatus), default=LiveStreamStatus.STARTING, nullable=False)
    visibility = Column(SQLEnum(LiveStreamVisibility), default=LiveStreamVisibility.PUBLIC, nullable=False)
    
    # Moderation
    is_approved = Column(Boolean, default=False, nullable=False)
    is_flagged = Column(Boolean, default=False, nullable=False)
    is_rejected = Column(Boolean, default=False, nullable=False)
    
    # Moderation details
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(UUID(as_uuid=True), nullable=True)
    flagged_at = Column(DateTime, nullable=True)
    flagged_by = Column(UUID(as_uuid=True), nullable=True)
    flag_reason = Column(Text, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    rejected_by = Column(UUID(as_uuid=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Engagement metrics
    current_viewers = Column(Integer, default=0, nullable=False)
    peak_viewers = Column(Integer, default=0, nullable=False)
    total_viewers = Column(Integer, default=0, nullable=False)  # Total unique viewers
    likes_count = Column(Integer, default=0, nullable=False)
    comments_count = Column(Integer, default=0, nullable=False)
    shares_count = Column(Integer, default=0, nullable=False)
    
    # Stream duration
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration = Column(Integer, default=0, nullable=False)  # Duration in seconds
    
    # Monetization
    is_monetized = Column(Boolean, default=False, nullable=False)
    ad_revenue = Column(Integer, default=0, nullable=False)  # Revenue in cents
    
    # Recording
    is_recorded = Column(Boolean, default=False, nullable=False)
    recording_url = Column(String(500), nullable=True)
    recording_duration = Column(Integer, default=0, nullable=False)  # Recording duration in seconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    owner_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Relationships
    owner = relationship("User", back_populates="live_streams")
    viewers = relationship("LiveStreamViewer", back_populates="live_stream", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<LiveStream(id={self.id}, title={self.title}, owner_id={self.owner_id})>"
    
    @property
    def is_live(self) -> bool:
        """Check if stream is currently live."""
        return self.status == LiveStreamStatus.LIVE
    
    @property
    def is_ended(self) -> bool:
        """Check if stream has ended."""
        return self.status == LiveStreamStatus.ENDED
    
    @property
    def is_public(self) -> bool:
        """Check if stream is publicly visible."""
        return self.visibility == LiveStreamVisibility.PUBLIC and self.is_approved
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.viewer_count == 0:
            return 0.0
        
        total_engagement = self.likes_count + self.comments_count + self.shares_count
        return (total_engagement / self.viewer_count) * 100
    
    def to_dict(self) -> dict:
        """Convert live stream to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "stream_key": self.stream_key,
            "channel_arn": self.channel_arn,
            "ingest_endpoint": self.ingest_endpoint,
            "playback_url": self.playback_url,
            "thumbnail_url": self.thumbnail_url,
            "preview_url": self.preview_url,
            "status": self.status.value,
            "visibility": self.visibility.value,
            "is_approved": self.is_approved,
            "is_flagged": self.is_flagged,
            "is_rejected": self.is_rejected,
            "viewer_count": self.viewer_count,
            "peak_viewer_count": self.peak_viewer_count,
            "total_views": self.total_views,
            "likes_count": self.likes_count,
            "comments_count": self.comments_count,
            "shares_count": self.shares_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration": self.duration,
            "is_monetized": self.is_monetized,
            "ad_revenue": self.ad_revenue,
            "is_recorded": self.is_recorded,
            "recording_url": self.recording_url,
            "recording_duration": self.recording_duration,
            "engagement_rate": self.engagement_rate,
            "owner_id": str(self.owner_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LiveStreamViewer(Base):
    """Live stream viewer model for tracking viewers."""
    
    __tablename__ = "live_stream_viewers"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    live_stream_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # Nullable for anonymous viewers
    
    # Viewer information
    session_id = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Viewing metrics
    joined_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    left_at = Column(DateTime, nullable=True)
    watch_duration = Column(Integer, default=0, nullable=False)  # Watch duration in seconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    live_stream = relationship("LiveStream", back_populates="viewers")
    
    def __repr__(self) -> str:
        return f"<LiveStreamViewer(id={self.id}, live_stream_id={self.live_stream_id}, user_id={self.user_id})>"
    
    def to_dict(self) -> dict:
        """Convert live stream viewer to dictionary."""
        return {
            "id": str(self.id),
            "live_stream_id": str(self.live_stream_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "left_at": self.left_at.isoformat() if self.left_at else None,
            "watch_duration": self.watch_duration,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
