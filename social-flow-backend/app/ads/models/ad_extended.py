"""
Extended Ad Models for Campaigns, Impressions, Clicks, and Revenue Sharing.

Enhanced models for comprehensive ad management and monetization.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


class BiddingType(str, Enum):
    """Bidding type for campaigns."""
    CPM = "cpm"  # Cost per mille (1000 impressions)
    CPC = "cpc"  # Cost per click
    CPA = "cpa"  # Cost per acquisition
    CPV = "cpv"  # Cost per view


class CampaignStatus(str, Enum):
    """Campaign status."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AdFormat(str, Enum):
    """Ad format types."""
    BANNER = "banner"
    VIDEO = "video"
    NATIVE = "native"
    SPONSORED_POST = "sponsored_post"
    PRE_ROLL = "pre_roll"
    MID_ROLL = "mid_roll"
    POST_ROLL = "post_roll"
    STORY = "story"


class AdCampaign(Base):
    """Ad campaign model for grouping ads."""
    
    __tablename__ = "ad_campaigns"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Campaign details
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    objective = Column(String(100), nullable=True)  # awareness, traffic, conversions
    
    # Advertiser
    advertiser_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Budget and bidding
    budget = Column(Float, nullable=False)
    daily_budget = Column(Float, nullable=True)
    bidding_type = Column(String(20), nullable=False, default=BiddingType.CPM)
    bid_amount = Column(Float, nullable=False)
    
    # Scheduling
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    
    # Status
    status = Column(String(20), nullable=False, default=CampaignStatus.DRAFT)
    is_paused = Column(Boolean, default=False, nullable=False)
    
    # Performance metrics
    total_impressions = Column(Integer, default=0, nullable=False)
    total_clicks = Column(Integer, default=0, nullable=False)
    total_conversions = Column(Integer, default=0, nullable=False)
    total_spend = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    advertiser = relationship("User", foreign_keys=[advertiser_id])
    creatives = relationship("AdCreative", back_populates="campaign", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_ad_campaigns_advertiser_id", "advertiser_id"),
        Index("idx_ad_campaigns_status", "status"),
        Index("idx_ad_campaigns_start_date", "start_date"),
    )


class AdCreative(Base):
    """Ad creative (the actual ad content)."""
    
    __tablename__ = "ad_creatives"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Campaign
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("ad_campaigns.id", ondelete="CASCADE"), nullable=False)
    
    # Creative details
    name = Column(String(200), nullable=False)
    format = Column(String(50), nullable=False)  # banner, video, native, etc.
    
    # Creative assets
    image_url = Column(String(500), nullable=True)
    video_url = Column(String(500), nullable=True)
    thumbnail_url = Column(String(500), nullable=True)
    headline = Column(String(200), nullable=True)
    body_text = Column(Text, nullable=True)
    call_to_action = Column(String(100), nullable=True)  # Learn More, Shop Now, etc.
    destination_url = Column(String(500), nullable=False)
    
    # Targeting (JSONB for flexible querying)
    targeting_criteria = Column(JSONB, nullable=True)  # age, gender, interests, location
    
    # Status
    is_approved = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Moderation
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(UUID(as_uuid=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Performance metrics
    impressions = Column(Integer, default=0, nullable=False)
    clicks = Column(Integer, default=0, nullable=False)
    conversions = Column(Integer, default=0, nullable=False)
    spend = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    campaign = relationship("AdCampaign", back_populates="creatives")
    impressions_log = relationship("AdImpression", back_populates="creative", cascade="all, delete-orphan")
    clicks_log = relationship("AdClick", back_populates="creative", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_ad_creatives_campaign_id", "campaign_id"),
        Index("idx_ad_creatives_format", "format"),
        Index("idx_ad_creatives_is_active", "is_active"),
    )


class AdImpression(Base):
    """Ad impression tracking."""
    
    __tablename__ = "ad_impressions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # References
    creative_id = Column(UUID(as_uuid=True), ForeignKey("ad_creatives.id", ondelete="CASCADE"), nullable=False)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("ad_campaigns.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Context
    placement = Column(String(100), nullable=True)  # feed, sidebar, video_player
    content_id = Column(UUID(as_uuid=True), nullable=True)  # Associated video/post ID
    content_type = Column(String(50), nullable=True)  # video, post, story
    
    # Tracking data
    session_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_type = Column(String(50), nullable=True)  # mobile, desktop, tablet
    platform = Column(String(50), nullable=True)  # ios, android, web
    
    # Geographic data
    country = Column(String(2), nullable=True)
    region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    
    # Viewability metrics
    is_viewable = Column(Boolean, default=False, nullable=False)
    view_duration = Column(Integer, nullable=True)  # milliseconds
    viewport_percentage = Column(Float, nullable=True)
    
    # Revenue
    cost = Column(Float, nullable=False)  # Cost charged to advertiser
    creator_revenue = Column(Float, default=0.0, nullable=False)  # Creator's share
    platform_revenue = Column(Float, default=0.0, nullable=False)  # Platform's share
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    creative = relationship("AdCreative", back_populates="impressions_log")
    
    __table_args__ = (
        Index("idx_ad_impressions_creative_id", "creative_id"),
        Index("idx_ad_impressions_campaign_id", "campaign_id"),
        Index("idx_ad_impressions_user_id", "user_id"),
        Index("idx_ad_impressions_created_at", "created_at"),
        Index("idx_ad_impressions_content_id", "content_id"),
    )


class AdClick(Base):
    """Ad click tracking."""
    
    __tablename__ = "ad_clicks"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # References
    impression_id = Column(UUID(as_uuid=True), nullable=True)  # Link to impression if available
    creative_id = Column(UUID(as_uuid=True), ForeignKey("ad_creatives.id", ondelete="CASCADE"), nullable=False)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("ad_campaigns.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Context
    placement = Column(String(100), nullable=True)
    content_id = Column(UUID(as_uuid=True), nullable=True)
    content_type = Column(String(50), nullable=True)
    
    # Tracking data
    session_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_type = Column(String(50), nullable=True)
    platform = Column(String(50), nullable=True)
    
    # Geographic data
    country = Column(String(2), nullable=True)
    region = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    
    # Click data
    click_position_x = Column(Integer, nullable=True)
    click_position_y = Column(Integer, nullable=True)
    
    # Revenue
    cost = Column(Float, nullable=False)  # Cost charged to advertiser
    creator_revenue = Column(Float, default=0.0, nullable=False)  # Creator's share
    platform_revenue = Column(Float, default=0.0, nullable=False)  # Platform's share
    
    # Fraud detection
    is_valid = Column(Boolean, default=True, nullable=False)
    fraud_score = Column(Float, default=0.0, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    creative = relationship("AdCreative", back_populates="clicks_log")
    
    __table_args__ = (
        Index("idx_ad_clicks_creative_id", "creative_id"),
        Index("idx_ad_clicks_campaign_id", "campaign_id"),
        Index("idx_ad_clicks_user_id", "user_id"),
        Index("idx_ad_clicks_created_at", "created_at"),
        Index("idx_ad_clicks_impression_id", "impression_id"),
    )


class AdCreatorRevenue(Base):
    """Track revenue sharing with content creators."""
    
    __tablename__ = "ad_creator_revenues"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # References
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content_id = Column(UUID(as_uuid=True), nullable=False)  # Video or post ID
    content_type = Column(String(50), nullable=False)  # video, post, story
    
    # Revenue details
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metrics
    total_impressions = Column(Integer, default=0, nullable=False)
    total_clicks = Column(Integer, default=0, nullable=False)
    
    # Revenue breakdown
    gross_revenue = Column(Float, default=0.0, nullable=False)  # Total ad revenue
    creator_share_percentage = Column(Float, default=70.0, nullable=False)  # Default 70%
    creator_revenue = Column(Float, default=0.0, nullable=False)  # Creator's earnings
    platform_revenue = Column(Float, default=0.0, nullable=False)  # Platform's cut
    
    # Payout status
    is_paid = Column(Boolean, default=False, nullable=False)
    paid_at = Column(DateTime, nullable=True)
    payout_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to creator_payouts
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    creator = relationship("User", foreign_keys=[creator_id])
    
    __table_args__ = (
        Index("idx_ad_creator_revenues_creator_id", "creator_id"),
        Index("idx_ad_creator_revenues_content_id", "content_id"),
        Index("idx_ad_creator_revenues_period_start", "period_start"),
        Index("idx_ad_creator_revenues_is_paid", "is_paid"),
    )
