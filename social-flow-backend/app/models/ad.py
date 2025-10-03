"""  
Advertising models for ad campaigns and targeting.

This module defines comprehensive advertising models including:
- Ad campaigns with budget management
- Ad creatives (video, image, text)
- Impressions and clicks tracking
- Advanced targeting (geo, demographics, interests, ML-based)
- Revenue tracking and reporting
"""

from __future__ import annotations

from datetime import datetime
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
    from app.models.video import Video


class AdType(str, PyEnum):
    """Ad creative type."""
    VIDEO = "video"
    IMAGE = "image"
    TEXT = "text"
    BANNER = "banner"
    NATIVE = "native"


class AdPlacement(str, PyEnum):
    """Ad placement location."""
    PRE_ROLL = "pre_roll"  # Before video
    MID_ROLL = "mid_roll"  # During video
    POST_ROLL = "post_roll"  # After video
    FEED = "feed"  # In social feed
    SIDEBAR = "sidebar"  # Sidebar banner
    OVERLAY = "overlay"  # Video overlay


class CampaignStatus(str, PyEnum):
    """Campaign status."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CampaignObjective(str, PyEnum):
    """Campaign objective."""
    BRAND_AWARENESS = "brand_awareness"
    REACH = "reach"
    TRAFFIC = "traffic"
    ENGAGEMENT = "engagement"
    APP_INSTALLS = "app_installs"
    VIDEO_VIEWS = "video_views"
    CONVERSIONS = "conversions"
    LEAD_GENERATION = "lead_generation"


class BidStrategy(str, PyEnum):
    """Bidding strategy."""
    CPM = "cpm"  # Cost per mille (thousand impressions)
    CPC = "cpc"  # Cost per click
    CPV = "cpv"  # Cost per view
    CPA = "cpa"  # Cost per action


class TargetingType(str, PyEnum):
    """Targeting type."""
    DEMOGRAPHIC = "demographic"
    GEOGRAPHIC = "geographic"
    INTEREST = "interest"
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"
    LOOKALIKE = "lookalike"
    RETARGETING = "retargeting"


class AdCampaign(CommonBase):
    """
    Ad campaign model.
    
    Represents an advertising campaign with budget, targeting, and scheduling.
    """
    
    __tablename__ = "ad_campaigns"
    
    # ==================== Basic Information ====================
    name = Column(
        String(200),
        nullable=False,
        doc="Campaign name"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Campaign description"
    )
    
    # ==================== Advertiser ====================
    advertiser_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Advertiser user ID"
    )
    
    # ==================== Status and Objective ====================
    status = Column(
        SQLEnum(CampaignStatus),
        default=CampaignStatus.DRAFT,
        nullable=False,
        index=True,
        doc="Campaign status"
    )
    
    objective = Column(
        SQLEnum(CampaignObjective),
        nullable=False,
        doc="Campaign objective"
    )
    
    # ==================== Budget ====================
    daily_budget = Column(
        Float,
        nullable=True,
        doc="Daily budget limit (USD)"
    )
    
    total_budget = Column(
        Float,
        nullable=True,
        doc="Total campaign budget (USD)"
    )
    
    total_spent = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Total amount spent (USD)"
    )
    
    # ==================== Bidding ====================
    bid_strategy = Column(
        SQLEnum(BidStrategy),
        nullable=False,
        doc="Bidding strategy"
    )
    
    bid_amount = Column(
        Float,
        nullable=False,
        doc="Bid amount (USD)"
    )
    
    # ==================== Scheduling ====================
    start_date = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Campaign start date"
    )
    
    end_date = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Campaign end date"
    )
    
    # ==================== Targeting ====================
    target_countries = Column(
        ARRAY(String(2)),
        default=[],
        nullable=False,
        doc="Target country codes (ISO 3166-1 alpha-2)"
    )
    
    target_age_min = Column(
        Integer,
        nullable=True,
        doc="Minimum target age"
    )
    
    target_age_max = Column(
        Integer,
        nullable=True,
        doc="Maximum target age"
    )
    
    target_genders = Column(
        ARRAY(String(20)),
        default=[],
        nullable=False,
        doc="Target genders (male, female, other)"
    )
    
    target_interests = Column(
        ARRAY(String(100)),
        default=[],
        nullable=False,
        doc="Target interest categories"
    )
    
    target_languages = Column(
        ARRAY(String(10)),
        default=[],
        nullable=False,
        doc="Target languages (ISO 639-1)"
    )
    
    custom_targeting = Column(
        JSONB,
        default={},
        nullable=False,
        doc="Custom targeting rules (ML-based, lookalike, etc.)"
    )
    
    # ==================== Performance Metrics ====================
    impressions = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total impressions"
    )
    
    clicks = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total clicks"
    )
    
    views = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total video views (for video ads)"
    )
    
    conversions = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total conversions"
    )
    
    # ==================== Calculated Metrics ====================
    ctr = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Click-through rate (clicks/impressions * 100)"
    )
    
    cpm = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Cost per mille (spent/impressions * 1000)"
    )
    
    cpc = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Cost per click (spent/clicks)"
    )
    
    cpa = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Cost per action (spent/conversions)"
    )
    
    # ==================== Relationships ====================
    advertiser: Mapped["User"] = relationship(
        "User",
        backref="ad_campaigns",
        foreign_keys=[advertiser_id]
    )
    
    ads: Mapped[list["Ad"]] = relationship(
        "Ad",
        back_populates="campaign",
        cascade="all, delete-orphan"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_campaign_advertiser_status', 'advertiser_id', 'status'),
        Index('idx_campaign_status_dates', 'status', 'start_date', 'end_date'),
        Index('idx_campaign_created', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<AdCampaign(id={self.id}, name={self.name}, status={self.status})>"


class Ad(CommonBase):
    """
    Ad creative model.
    
    Represents an individual ad creative within a campaign.
    """
    
    __tablename__ = "ads"
    
    # ==================== Campaign ====================
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey('ad_campaigns.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Parent campaign ID"
    )
    
    # ==================== Basic Information ====================
    name = Column(
        String(200),
        nullable=False,
        doc="Ad name"
    )
    
    ad_type = Column(
        SQLEnum(AdType),
        nullable=False,
        doc="Ad creative type"
    )
    
    placement = Column(
        SQLEnum(AdPlacement),
        nullable=False,
        doc="Ad placement"
    )
    
    # ==================== Creative Content ====================
    headline = Column(
        String(100),
        nullable=True,
        doc="Ad headline"
    )
    
    body_text = Column(
        Text,
        nullable=True,
        doc="Ad body text"
    )
    
    call_to_action = Column(
        String(50),
        nullable=True,
        doc="Call to action text"
    )
    
    # ==================== Media ====================
    image_url = Column(
        String(500),
        nullable=True,
        doc="Image URL for image/banner ads"
    )
    
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey('videos.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Video ID for video ads"
    )
    
    video_url = Column(
        String(500),
        nullable=True,
        doc="External video URL"
    )
    
    thumbnail_url = Column(
        String(500),
        nullable=True,
        doc="Thumbnail URL"
    )
    
    # ==================== Destination ====================
    destination_url = Column(
        String(500),
        nullable=False,
        doc="Click destination URL"
    )
    
    tracking_params = Column(
        JSONB,
        default={},
        nullable=False,
        doc="URL tracking parameters"
    )
    
    # ==================== Status ====================
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether ad is active"
    )
    
    is_approved = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether ad passed moderation"
    )
    
    approval_notes = Column(
        Text,
        nullable=True,
        doc="Moderation notes"
    )
    
    # ==================== Performance Metrics ====================
    impressions = Column(
        BigInteger,
        default=0,
        nullable=False,
        index=True,
        doc="Total impressions"
    )
    
    clicks = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total clicks"
    )
    
    views = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total video views"
    )
    
    conversions = Column(
        BigInteger,
        default=0,
        nullable=False,
        doc="Total conversions"
    )
    
    total_spent = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Total spent on this ad (USD)"
    )
    
    # ==================== Relationships ====================
    campaign: Mapped["AdCampaign"] = relationship(
        "AdCampaign",
        back_populates="ads"
    )
    
    video: Mapped["Video"] = relationship(
        "Video",
        backref="ads"
    )
    
    impressions_records: Mapped[list["AdImpression"]] = relationship(
        "AdImpression",
        back_populates="ad",
        cascade="all, delete-orphan"
    )
    
    clicks_records: Mapped[list["AdClick"]] = relationship(
        "AdClick",
        back_populates="ad",
        cascade="all, delete-orphan"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_ad_campaign_active', 'campaign_id', 'is_active'),
        Index('idx_ad_type_placement', 'ad_type', 'placement'),
        Index('idx_ad_impressions', 'impressions'),
        Index('idx_ad_created', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Ad(id={self.id}, name={self.name}, campaign_id={self.campaign_id})>"


class AdImpression(CommonBase):
    """
    Ad impression tracking model.
    
    Records each time an ad is displayed to a user.
    High-volume time-series data.
    """
    
    __tablename__ = "ad_impressions"
    
    # ==================== Ad ====================
    ad_id = Column(
        UUID(as_uuid=True),
        ForeignKey('ads.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Ad ID"
    )
    
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey('ad_campaigns.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Campaign ID (denormalized)"
    )
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="User ID (NULL for anonymous)"
    )
    
    session_id = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Session ID for tracking unique impressions"
    )
    
    # ==================== Context ====================
    placement = Column(
        SQLEnum(AdPlacement),
        nullable=False,
        doc="Where ad was shown"
    )
    
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey('videos.id', ondelete='SET NULL'),
        nullable=True,
        doc="Video where ad was shown (if applicable)"
    )
    
    post_id = Column(
        UUID(as_uuid=True),
        ForeignKey('posts.id', ondelete='SET NULL'),
        nullable=True,
        doc="Post where ad was shown (if applicable)"
    )
    
    # ==================== Viewability ====================
    is_viewable = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether ad was actually viewable (50% visible for 1s)"
    )
    
    view_duration = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="How long ad was visible (seconds)"
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
        doc="Country code (ISO 3166-1)"
    )
    
    city = Column(
        String(100),
        nullable=True,
        doc="City name"
    )
    
    # ==================== Device Data ====================
    user_agent = Column(
        Text,
        nullable=True,
        doc="User agent string"
    )
    
    device_type = Column(
        String(50),
        nullable=True,
        doc="Device type: mobile, tablet, desktop"
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
    
    # ==================== Revenue ====================
    cost = Column(
        Float,
        nullable=False,
        doc="Cost of this impression (USD)"
    )
    
    # ==================== Relationships ====================
    ad: Mapped["Ad"] = relationship(
        "Ad",
        back_populates="impressions_records"
    )
    
    campaign: Mapped["AdCampaign"] = relationship("AdCampaign")
    user: Mapped["User"] = relationship("User")
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_impression_ad_created', 'ad_id', 'created_at'),
        Index('idx_impression_campaign_created', 'campaign_id', 'created_at'),
        Index('idx_impression_user_created', 'user_id', 'created_at'),
        Index('idx_impression_session', 'session_id', 'ad_id'),
        Index('idx_impression_country', 'country_code', 'created_at'),
        
        # Partition by created_at (daily for high volume)
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<AdImpression(id={self.id}, ad_id={self.ad_id})>"


class AdClick(CommonBase):
    """
    Ad click tracking model.
    
    Records each time a user clicks on an ad.
    """
    
    __tablename__ = "ad_clicks"
    
    # ==================== Ad ====================
    ad_id = Column(
        UUID(as_uuid=True),
        ForeignKey('ads.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Ad ID"
    )
    
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey('ad_campaigns.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Campaign ID (denormalized)"
    )
    
    impression_id = Column(
        UUID(as_uuid=True),
        ForeignKey('ad_impressions.id', ondelete='SET NULL'),
        nullable=True,
        doc="Related impression ID"
    )
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="User ID (NULL for anonymous)"
    )
    
    session_id = Column(
        String(255),
        nullable=False,
        index=True,
        doc="Session ID"
    )
    
    # ==================== Context ====================
    placement = Column(
        SQLEnum(AdPlacement),
        nullable=False,
        doc="Where ad was clicked"
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
    
    # ==================== Conversion Tracking ====================
    converted = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether click resulted in conversion"
    )
    
    converted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Conversion timestamp"
    )
    
    conversion_value = Column(
        Float,
        nullable=True,
        doc="Conversion value (USD)"
    )
    
    # ==================== Revenue ====================
    cost = Column(
        Float,
        nullable=False,
        doc="Cost of this click (USD)"
    )
    
    # ==================== Relationships ====================
    ad: Mapped["Ad"] = relationship(
        "Ad",
        back_populates="clicks_records"
    )
    
    campaign: Mapped["AdCampaign"] = relationship("AdCampaign")
    impression: Mapped["AdImpression"] = relationship("AdImpression")
    user: Mapped["User"] = relationship("User")
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_click_ad_created', 'ad_id', 'created_at'),
        Index('idx_click_campaign_created', 'campaign_id', 'created_at'),
        Index('idx_click_user_created', 'user_id', 'created_at'),
        Index('idx_click_converted', 'converted', 'created_at'),
        Index('idx_click_country', 'country_code', 'created_at'),
        
        # Partition by created_at
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<AdClick(id={self.id}, ad_id={self.ad_id})>"


# Export models
__all__ = [
    'AdCampaign',
    'Ad',
    'AdImpression',
    'AdClick',
    'AdType',
    'AdPlacement',
    'CampaignStatus',
    'CampaignObjective',
    'BidStrategy',
    'TargetingType',
]
