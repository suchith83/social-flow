"""
Live Streaming Models

Defines database models for live streaming functionality including:
- LiveStream: Core streaming sessions
- StreamViewer: Active viewer tracking
- ChatMessage: Live chat messages
- StreamRecording: Archived stream recordings
"""

from datetime import datetime
import secrets
from enum import Enum
from uuid import uuid4

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, 
    ForeignKey, Text, Float, JSON, Index
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class StreamStatus(str, Enum):
    """Stream status enumeration"""
    SCHEDULED = "scheduled"
    STARTING = "starting"
    LIVE = "live"
    ENDING = "ending"
    ENDED = "ended"
    FAILED = "failed"


class StreamQuality(str, Enum):
    """Stream quality enumeration"""
    LOW = "low"  # 480p
    MEDIUM = "medium"  # 720p
    HIGH = "high"  # 1080p
    ULTRA = "ultra"  # 4K


class ChatMessageType(str, Enum):
    """Chat message type enumeration"""
    MESSAGE = "message"
    SYSTEM = "system"
    DONATION = "donation"
    SUBSCRIPTION = "subscription"
    MODERATOR_ACTION = "moderator_action"


class LiveStream(Base):
    """
    Live streaming session model
    
    Manages RTMP streams with AWS MediaLive/IVS integration
    """
    __tablename__ = "live_streams"

    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Relationships
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Stream configuration
    stream_key = Column(String(64), unique=True, nullable=False, index=True, default=lambda: secrets.token_urlsafe(32))
    stream_url = Column(String(512))  # RTMP ingest URL
    playback_url = Column(String(512))  # HLS/DASH playback URL
    
    # Stream metadata
    title = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    tags = Column(JSON, default=list)
    thumbnail_url = Column(String(512))
    
    # Stream status
    status = Column(String(20), nullable=False, default=StreamStatus.SCHEDULED.value, index=True)
    quality = Column(String(20), default=StreamQuality.HIGH.value)
    
    # Streaming details
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    scheduled_start = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer, default=0)
    
    # Viewer metrics
    current_viewers = Column(Integer, default=0)
    peak_viewers = Column(Integer, default=0)
    total_views = Column(Integer, default=0)
    
    # Engagement metrics
    likes_count = Column(Integer, default=0)
    chat_messages_count = Column(Integer, default=0)
    
    # Monetization
    is_monetized = Column(Boolean, default=False)
    subscription_only = Column(Boolean, default=False)
    donation_enabled = Column(Boolean, default=True)
    total_revenue = Column(Float, default=0.0)
    
    # Recording
    is_recording = Column(Boolean, default=True)
    recording_url = Column(String(512))
    recording_bucket = Column(String(100))
    recording_key = Column(String(512))
    
    # AWS MediaLive/IVS
    channel_id = Column(String(100))  # MediaLive channel ID
    ivs_channel_arn = Column(String(200))  # IVS channel ARN
    ivs_playback_url = Column(String(512))
    ivs_ingest_endpoint = Column(String(512))
    
    # Moderation
    chat_enabled = Column(Boolean, default=True)
    chat_delay_seconds = Column(Integer, default=0)
    slow_mode_enabled = Column(Boolean, default=False)
    slow_mode_interval = Column(Integer, default=3)
    
    # Privacy
    is_public = Column(Boolean, default=True)
    is_unlisted = Column(Boolean, default=False)
    
    # Technical metrics
    bitrate_kbps = Column(Integer)
    framerate_fps = Column(Integer)
    resolution = Column(String(20))
    codec = Column(String(20))
    
    # Metadata
    stream_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="live_streams")
    # Monetization relationships are optional and defined in their modules
    viewers = relationship("StreamViewer", back_populates="stream", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="stream", cascade="all, delete-orphan")
    recordings = relationship("StreamRecording", back_populates="stream", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_stream_user_status", "user_id", "status"),
        Index("idx_stream_status_scheduled", "status", "scheduled_start"),
        Index("idx_stream_created", "created_at"),
        Index("idx_stream_public", "is_public", "status"),
        {'extend_existing': True}
    )
    
    @property
    def is_live(self) -> bool:
        """Check if stream is currently live"""
        return self.status == StreamStatus.LIVE.value
    
    @property
    def is_scheduled(self) -> bool:
        """Check if stream is scheduled"""
        return self.status == StreamStatus.SCHEDULED.value
    
    @property
    def duration_minutes(self) -> int:
        """Get duration in minutes"""
        return self.duration_seconds // 60 if self.duration_seconds else 0
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate"""
        if self.total_views == 0:
            return 0.0
        return (self.likes_count + self.chat_messages_count) / self.total_views
    
    def __repr__(self):
        return f"<LiveStream {self.id} - {self.title} ({self.status})>"

    def __init__(self, **kwargs):
        # Map test aliases to model fields
        if 'is_private' in kwargs:
            is_private = kwargs.pop('is_private')
            # If explicitly provided, invert to set is_public
            kwargs['is_public'] = not bool(is_private)
        if 'record_stream' in kwargs:
            record_stream = kwargs.pop('record_stream')
            kwargs['is_recording'] = bool(record_stream)
        # Ensure tags is a list if provided as None
        if 'tags' in kwargs and kwargs['tags'] is None:
            kwargs['tags'] = []
        super().__init__(**kwargs)


class StreamViewer(Base):
    """
    Stream viewer tracking model
    
    Tracks active viewers in real-time
    """
    __tablename__ = "stream_viewers"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Relationships
    stream_id = Column(PGUUID(as_uuid=True), ForeignKey("live_streams.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    
    # Viewer info
    session_id = Column(String(100), nullable=False, index=True)
    ip_address = Column(String(45))  # IPv6 support
    user_agent = Column(String(512))
    
    # Viewing session
    joined_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    last_heartbeat = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    left_at = Column(DateTime(timezone=True))
    watch_time_seconds = Column(Integer, default=0)
    
    # Quality metrics
    selected_quality = Column(String(20), default=StreamQuality.HIGH.value)
    buffer_count = Column(Integer, default=0)
    
    # Metadata
    viewer_metadata = Column(JSON, default=dict)
    
    # Relationships
    stream = relationship("LiveStream", back_populates="viewers")
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("idx_viewer_stream_active", "stream_id", "left_at"),
        Index("idx_viewer_user", "user_id"),
        Index("idx_viewer_heartbeat", "last_heartbeat"),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if viewer is currently active (heartbeat within 30s)"""
        if self.left_at:
            return False
        delta = datetime.utcnow() - self.last_heartbeat
        return delta.total_seconds() < 30
    
    @property
    def watch_time_minutes(self) -> int:
        """Get watch time in minutes"""
        return self.watch_time_seconds // 60 if self.watch_time_seconds else 0
    
    def __repr__(self):
        return f"<StreamViewer {self.id} - Stream {self.stream_id}>"

    def __init__(self, **kwargs):
        # Accept and ignore 'is_active' passed by some tests
        kwargs.pop('is_active', None)
        # Ensure a session_id exists
        if not kwargs.get('session_id'):
            kwargs['session_id'] = secrets.token_urlsafe(16)
        super().__init__(**kwargs)


class ChatMessage(Base):
    """
    Live stream chat message model
    
    Stores all chat messages with moderation support
    """
    __tablename__ = "chat_messages"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Relationships
    stream_id = Column(PGUUID(as_uuid=True), ForeignKey("live_streams.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Message content
    message_type = Column(String(30), nullable=False, default=ChatMessageType.MESSAGE.value)
    content = Column(Text, nullable=False)
    
    # Message metadata
    sent_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    edited_at = Column(DateTime(timezone=True))
    
    # Moderation
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime(timezone=True))
    deleted_by_user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    deletion_reason = Column(String(200))
    
    # Toxicity detection
    is_flagged = Column(Boolean, default=False)
    toxicity_score = Column(Float)
    flagged_reason = Column(String(200))
    
    # Donation/Subscription
    donation_amount = Column(Float)
    donation_currency = Column(String(3))
    
    # Metadata
    message_metadata = Column(JSON, default=dict)
    
    # Relationships
    stream = relationship("LiveStream", back_populates="chat_messages")
    user = relationship("User", foreign_keys=[user_id])
    deleted_by = relationship("User", foreign_keys=[deleted_by_user_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_chat_stream_time", "stream_id", "sent_at"),
        Index("idx_chat_user", "user_id"),
        Index("idx_chat_flagged", "is_flagged"),
        Index("idx_chat_deleted", "is_deleted"),
    )
    
    @property
    def is_system_message(self) -> bool:
        """Check if message is system-generated"""
        return self.message_type != ChatMessageType.MESSAGE.value
    
    @property
    def is_donation_message(self) -> bool:
        """Check if message is a donation"""
        return self.message_type == ChatMessageType.DONATION.value
    
    def __repr__(self):
        return f"<ChatMessage {self.id} - Stream {self.stream_id}>"


class StreamRecording(Base):
    """
    Stream recording model
    
    Stores archived stream recordings with metadata
    """
    __tablename__ = "stream_recordings"
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Relationships
    stream_id = Column(PGUUID(as_uuid=True), ForeignKey("live_streams.id", ondelete="CASCADE"), nullable=False)
    
    # Recording details
    recording_url = Column(String(512), nullable=False)
    bucket_name = Column(String(100), nullable=False)
    object_key = Column(String(512), nullable=False)
    
    # Recording metadata
    duration_seconds = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer)
    format = Column(String(20), default="mp4")
    resolution = Column(String(20))
    bitrate_kbps = Column(Integer)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    
    # Availability
    is_available = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True))
    
    # Metadata
    recording_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Relationships
    stream = relationship("LiveStream", back_populates="recordings")
    
    # Indexes
    __table_args__ = (
        Index("idx_recording_stream", "stream_id"),
        Index("idx_recording_available", "is_available"),
        Index("idx_recording_created", "created_at"),
    )
    
    @property
    def duration_minutes(self) -> int:
        """Get duration in minutes"""
        return self.duration_seconds // 60 if self.duration_seconds else 0
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB"""
        return self.file_size_bytes / (1024 * 1024) if self.file_size_bytes else 0.0
    
    def __repr__(self):
        return f"<StreamRecording {self.id} - Stream {self.stream_id}>"
