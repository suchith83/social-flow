"""  
Live streaming models for AWS IVS integration.

This module defines comprehensive live streaming models including:
- Live streams with AWS IVS
- Stream chat (real-time messaging)
- Stream donations (tips during live)
- Viewer tracking
- Recording management
"""

from __future__ import annotations

from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Enum as SQLEnum,
    Float, ForeignKey, Index, Integer, String, Text
)
from sqlalchemy.orm import Mapped, relationship

from app.models.base import CommonBase
from app.models.types import ARRAY, JSONB, UUID

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.payment import Payment


class StreamStatus(str, PyEnum):
    """Live stream status."""
    SCHEDULED = "scheduled"
    STARTING = "starting"
    LIVE = "live"
    PAUSED = "paused"
    ENDED = "ended"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StreamVisibility(str, PyEnum):
    """Stream visibility level."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    SUBSCRIBERS_ONLY = "subscribers_only"
    PRIVATE = "private"


class StreamQuality(str, PyEnum):
    """Stream quality preset."""
    LOW = "low"  # 480p
    MEDIUM = "medium"  # 720p
    HIGH = "high"  # 1080p
    ULTRA = "ultra"  # 4K


class DonationStatus(str, PyEnum):
    """Donation status."""
    PENDING = "pending"
    COMPLETED = "completed"
    REFUNDED = "refunded"
    FAILED = "failed"


class LiveStream(CommonBase):
    """
    Live stream model for AWS IVS integration.
    
    Manages live streaming sessions with real-time viewer tracking,
    chat, donations, and recording.
    """
    
    __tablename__ = "live_streams"
    
    # ==================== Basic Information ====================
    title = Column(
        String(200),
        nullable=False,
        index=True,
        doc="Stream title"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Stream description"
    )
    
    category = Column(
        String(50),
        nullable=True,
        index=True,
        doc="Stream category (gaming, music, education, etc.)"
    )
    
    tags = Column(
        ARRAY(String(50)),
        default=[],
        nullable=False,
        doc="Stream tags"
    )
    
    # ==================== Streamer ====================
    streamer_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Streamer user ID"
    )
    
    # ==================== AWS IVS ====================
    ivs_channel_arn = Column(
        String(500),
        unique=True,
        nullable=True,
        index=True,
        doc="AWS IVS Channel ARN"
    )
    
    ivs_stream_key_arn = Column(
        String(500),
        nullable=True,
        doc="AWS IVS Stream Key ARN"
    )
    
    ivs_ingest_endpoint = Column(
        String(500),
        nullable=True,
        doc="RTMP ingest endpoint"
    )
    
    ivs_playback_url = Column(
        String(500),
        nullable=True,
        doc="HLS playback URL"
    )
    
    ivs_stream_session_id = Column(
        String(255),
        nullable=True,
        doc="Current IVS stream session ID"
    )
    
    # ==================== Streaming Configuration ====================
    stream_quality = Column(
        SQLEnum(StreamQuality),
        default=StreamQuality.HIGH,
        nullable=False,
        doc="Stream quality preset"
    )
    
    enable_low_latency = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable low-latency mode"
    )
    
    enable_recording = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable automatic recording"
    )
    
    enable_chat = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable live chat"
    )
    
    enable_donations = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable donations during stream"
    )
    
    # ==================== Status ====================
    status = Column(
        SQLEnum(StreamStatus),
        default=StreamStatus.SCHEDULED,
        nullable=False,
        index=True,
        doc="Stream status"
    )
    
    visibility = Column(
        SQLEnum(StreamVisibility),
        default=StreamVisibility.PUBLIC,
        nullable=False,
        index=True,
        doc="Stream visibility"
    )
    
    # ==================== Scheduling ====================
    scheduled_start_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Scheduled start time"
    )
    
    actual_start_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Actual start time"
    )
    
    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Stream end time"
    )
    
    # ==================== Duration ====================
    duration = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Stream duration in seconds"
    )
    
    # ==================== Thumbnails ====================
    thumbnail_url = Column(
        String(500),
        nullable=True,
        doc="Stream thumbnail URL"
    )
    
    preview_thumbnail_url = Column(
        String(500),
        nullable=True,
        doc="Live preview thumbnail URL (auto-generated)"
    )
    
    # ==================== Recording ====================
    recording_enabled = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether recording is enabled"
    )
    
    recording_s3_bucket = Column(
        String(100),
        nullable=True,
        doc="S3 bucket for recording"
    )
    
    recording_s3_key = Column(
        String(500),
        nullable=True,
        doc="S3 key for recording"
    )
    
    recording_url = Column(
        String(500),
        nullable=True,
        doc="Recording playback URL (after stream ends)"
    )
    
    recording_duration = Column(
        Integer,
        nullable=True,
        doc="Recording duration in seconds"
    )
    
    # ==================== Viewer Metrics ====================
    current_viewers = Column(
        Integer,
        default=0,
        nullable=False,
        index=True,
        doc="Current viewer count (real-time)"
    )
    
    peak_viewers = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Peak concurrent viewers"
    )
    
    total_views = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total views (all time)"
    )
    
    unique_viewers = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Unique viewer count"
    )
    
    average_watch_time = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Average watch time in seconds"
    )
    
    # ==================== Engagement ====================
    like_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total likes"
    )
    
    chat_message_count = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total chat messages"
    )
    
    donation_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Total donations received"
    )
    
    total_donations_amount = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Total donation amount (USD)"
    )
    
    # ==================== Monetization ====================
    is_monetized = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether stream is monetized"
    )
    
    ad_breaks = Column(
        JSONB,
        default=[],
        nullable=False,
        doc="Ad break timestamps"
    )
    
    total_revenue = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Total revenue (ads + donations)"
    )
    
    # ==================== Moderation ====================
    is_mature_content = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="18+ mature content flag"
    )
    
    banned_words = Column(
        ARRAY(String(100)),
        default=[],
        nullable=False,
        doc="Chat banned words"
    )
    
    moderator_ids = Column(
        ARRAY(UUID(as_uuid=True)),
        default=[],
        nullable=False,
        doc="Chat moderator user IDs"
    )
    
    # ==================== Relationships ====================
    streamer: Mapped["User"] = relationship(
        "User",
        backref="live_streams",
        foreign_keys=[streamer_id]
    )
    
    chat_messages: Mapped[list["StreamChat"]] = relationship(
        "StreamChat",
        back_populates="stream",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    donations: Mapped[list["StreamDonation"]] = relationship(
        "StreamDonation",
        back_populates="stream",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    viewers: Mapped[list["StreamViewer"]] = relationship(
        "StreamViewer",
        back_populates="stream",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_stream_streamer_status', 'streamer_id', 'status'),
        Index('idx_stream_status_visibility', 'status', 'visibility'),
        Index('idx_stream_scheduled', 'scheduled_start_at'),
        Index('idx_stream_live', 'status', 'current_viewers'),
        Index('idx_stream_category', 'category', 'status'),
        Index('idx_stream_created', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<LiveStream(id={self.id}, title={self.title}, status={self.status})>"
    
    def is_live(self) -> bool:
        """Check if stream is currently live."""
        return self.status == StreamStatus.LIVE


class StreamChat(CommonBase):
    """
    Stream chat message model.
    
    Real-time chat messages during live streams.
    High-volume time-series data.
    """
    
    __tablename__ = "stream_chat"
    
    # ==================== Stream ====================
    stream_id = Column(
        UUID(as_uuid=True),
        ForeignKey('live_streams.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Stream ID"
    )
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User ID"
    )
    
    # ==================== Message ====================
    message = Column(
        Text,
        nullable=False,
        doc="Chat message content"
    )
    
    # ==================== Message Type ====================
    is_moderator = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether user is a moderator"
    )
    
    is_subscriber = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether user is a subscriber"
    )
    
    is_pinned = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether message is pinned"
    )
    
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether message was deleted"
    )
    
    # ==================== Moderation ====================
    deleted_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        doc="Moderator who deleted the message"
    )
    
    deleted_reason = Column(
        Text,
        nullable=True,
        doc="Deletion reason"
    )
    
    # ==================== Relationships ====================
    stream: Mapped["LiveStream"] = relationship(
        "LiveStream",
        back_populates="chat_messages"
    )
    
    user: Mapped["User"] = relationship(
        "User",
        foreign_keys=[user_id],
        backref="stream_chat_messages"
    )
    
    deleted_by: Mapped["User"] = relationship(
        "User",
        foreign_keys=[deleted_by_id]
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_chat_stream_created', 'stream_id', 'created_at'),
        Index('idx_chat_user_created', 'user_id', 'created_at'),
        Index('idx_chat_pinned', 'stream_id', 'is_pinned'),
        
        # Partition by created_at
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<StreamChat(id={self.id}, stream_id={self.stream_id}, user_id={self.user_id})>"


class StreamDonation(CommonBase):
    """
    Stream donation model.
    
    Tracks donations/tips during live streams.
    """
    
    __tablename__ = "stream_donations"
    
    # ==================== Stream ====================
    stream_id = Column(
        UUID(as_uuid=True),
        ForeignKey('live_streams.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Stream ID"
    )
    
    # ==================== Donor ====================
    donor_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Donor user ID (NULL for anonymous)"
    )
    
    donor_name = Column(
        String(100),
        nullable=True,
        doc="Donor display name (for anonymous donors)"
    )
    
    # ==================== Amount ====================
    amount = Column(
        Float,
        nullable=False,
        doc="Donation amount (USD)"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Currency code"
    )
    
    # ==================== Message ====================
    message = Column(
        Text,
        nullable=True,
        doc="Donation message"
    )
    
    show_on_stream = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether to show donation on stream overlay"
    )
    
    # ==================== Status ====================
    status = Column(
        SQLEnum(DonationStatus),
        default=DonationStatus.PENDING,
        nullable=False,
        index=True,
        doc="Donation status"
    )
    
    # ==================== Payment ====================
    payment_id = Column(
        UUID(as_uuid=True),
        ForeignKey('payments.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Related payment ID"
    )
    
    stripe_payment_intent_id = Column(
        String(255),
        nullable=True,
        doc="Stripe payment intent ID"
    )
    
    # ==================== Processing ====================
    processing_fee = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Payment processing fee"
    )
    
    platform_fee = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Platform fee"
    )
    
    net_amount = Column(
        Float,
        nullable=False,
        doc="Net amount to streamer"
    )
    
    # ==================== Timestamps ====================
    processed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Processing completion timestamp"
    )
    
    refunded_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Refund timestamp"
    )
    
    # ==================== Relationships ====================
    stream: Mapped["LiveStream"] = relationship(
        "LiveStream",
        back_populates="donations"
    )
    
    donor: Mapped["User"] = relationship(
        "User",
        backref="stream_donations",
        foreign_keys=[donor_id]
    )
    
    payment: Mapped["Payment"] = relationship("Payment")
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_donation_stream_created', 'stream_id', 'created_at'),
        Index('idx_donation_donor_created', 'donor_id', 'created_at'),
        Index('idx_donation_status', 'status', 'created_at'),
        Index('idx_donation_amount', 'amount', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<StreamDonation(id={self.id}, amount={self.amount}, stream_id={self.stream_id})>"


class StreamViewer(CommonBase):
    """
    Stream viewer tracking model.
    
    Tracks individual viewer sessions for analytics.
    """
    
    __tablename__ = "stream_viewers"
    
    # ==================== Stream ====================
    stream_id = Column(
        UUID(as_uuid=True),
        ForeignKey('live_streams.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Stream ID"
    )
    
    # ==================== Viewer ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Viewer user ID (NULL for anonymous)"
    )
    
    session_id = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Viewer session ID"
    )
    
    # ==================== Viewing Session ====================
    joined_at = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Session start timestamp"
    )
    
    left_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Session end timestamp"
    )
    
    watch_duration = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Total watch time in seconds"
    )
    
    # ==================== Geographic Data ====================
    ip_address = Column(
        String(45),
        nullable=True,
        doc="IP address (anonymized)"
    )
    
    country_code = Column(
        String(2),
        nullable=True,
        index=True,
        doc="Country code"
    )
    
    city = Column(
        String(100),
        nullable=True,
        doc="City name"
    )
    
    # ==================== Device Data ====================
    device_type = Column(
        String(50),
        nullable=True,
        doc="Device type"
    )
    
    browser = Column(
        String(50),
        nullable=True,
        doc="Browser name"
    )
    
    os = Column(
        String(50),
        nullable=True,
        doc="Operating system"
    )
    
    # ==================== Engagement ====================
    chat_messages_sent = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of chat messages sent"
    )
    
    donated = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether viewer donated"
    )
    
    # ==================== Relationships ====================
    stream: Mapped["LiveStream"] = relationship(
        "LiveStream",
        back_populates="viewers"
    )
    
    user: Mapped["User"] = relationship("User")
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_viewer_stream_joined', 'stream_id', 'joined_at'),
        Index('idx_viewer_user_joined', 'user_id', 'joined_at'),
        Index('idx_viewer_session', 'session_id', 'stream_id'),
        Index('idx_viewer_country', 'country_code', 'joined_at'),
        
        # Partition by created_at
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<StreamViewer(id={self.id}, stream_id={self.stream_id})>"


# Export models
__all__ = [
    'LiveStream',
    'StreamChat',
    'StreamDonation',
    'StreamViewer',
    'StreamStatus',
    'StreamVisibility',
    'StreamQuality',
    'DonationStatus',
]
