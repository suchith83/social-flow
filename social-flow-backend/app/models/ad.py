"""
Ad model and related schemas.

This module defines the Ad model and related Pydantic schemas
for advertisement management in the Social Flow backend.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class AdType(str, Enum):
    """Advertisement type."""
    BANNER = "banner"
    VIDEO = "video"
    SPONSORED_POST = "sponsored_post"
    PRE_ROLL = "pre_roll"
    MID_ROLL = "mid_roll"
    POST_ROLL = "post_roll"


class AdStatus(str, Enum):
    """Advertisement status."""
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAUSED = "paused"
    ACTIVE = "active"
    EXPIRED = "expired"


class Ad(Base):
    """Ad model for storing advertisement information."""
    
    __tablename__ = "ads"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    ad_type = Column(String(50), nullable=False)  # banner, video, sponsored_post, etc.
    
    # Creative assets
    image_url = Column(String(500), nullable=True)
    video_url = Column(String(500), nullable=True)
    thumbnail_url = Column(String(500), nullable=True)
    
    # Targeting
    target_audience = Column(Text, nullable=True)  # JSON string of targeting criteria
    target_keywords = Column(Text, nullable=True)  # JSON string of keywords
    target_hashtags = Column(Text, nullable=True)  # JSON string of hashtags
    target_demographics = Column(Text, nullable=True)  # JSON string of demographics
    
    # Campaign information
    campaign_id = Column(String(100), nullable=True)
    ad_group_id = Column(String(100), nullable=True)
    
    # Budget and bidding
    budget = Column(Float, nullable=False)
    bid_amount = Column(Float, nullable=False)
    daily_budget = Column(Float, nullable=True)
    
    # Performance metrics
    impressions = Column(Integer, default=0, nullable=False)
    clicks = Column(Integer, default=0, nullable=False)
    views = Column(Integer, default=0, nullable=False)
    conversions = Column(Integer, default=0, nullable=False)
    
    # Revenue metrics
    cost = Column(Float, default=0.0, nullable=False)
    revenue = Column(Float, default=0.0, nullable=False)
    ctr = Column(Float, default=0.0, nullable=False)  # Click-through rate
    cpm = Column(Float, default=0.0, nullable=False)  # Cost per mille
    cpc = Column(Float, default=0.0, nullable=False)  # Cost per click
    cpa = Column(Float, default=0.0, nullable=False)  # Cost per acquisition
    
    # Status and moderation
    status = Column(String(20), default=AdStatus.DRAFT, nullable=False)
    is_approved = Column(Boolean, default=False, nullable=False)
    is_paused = Column(Boolean, default=False, nullable=False)
    
    # Moderation details
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(UUID(as_uuid=True), nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    rejected_by = Column(UUID(as_uuid=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    paused_by = Column(UUID(as_uuid=True), nullable=True)
    
    # Scheduling
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    advertiser_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Relationships
    advertiser = relationship("User", foreign_keys=[advertiser_id])
    
    def __repr__(self) -> str:
        return f"<Ad(id={self.id}, title={self.title}, advertiser_id={self.advertiser_id})>"
    
    @property
    def is_active(self) -> bool:
        """Check if ad is currently active."""
        if not self.is_approved or self.is_paused:
            return False
        
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        
        return True
    
    @property
    def click_through_rate(self) -> float:
        """Calculate click-through rate."""
        if self.impressions == 0:
            return 0.0
        
        return (self.clicks / self.impressions) * 100
    
    @property
    def cost_per_mille(self) -> float:
        """Calculate cost per mille."""
        if self.impressions == 0:
            return 0.0
        
        return (self.cost / self.impressions) * 1000
    
    @property
    def cost_per_click(self) -> float:
        """Calculate cost per click."""
        if self.clicks == 0:
            return 0.0
        
        return self.cost / self.clicks
    
    @property
    def cost_per_acquisition(self) -> float:
        """Calculate cost per acquisition."""
        if self.conversions == 0:
            return 0.0
        
        return self.cost / self.conversions
    
    def to_dict(self) -> dict:
        """Convert ad to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "ad_type": self.ad_type,
            "image_url": self.image_url,
            "video_url": self.video_url,
            "thumbnail_url": self.thumbnail_url,
            "target_audience": self.target_audience,
            "target_keywords": self.target_keywords,
            "target_hashtags": self.target_hashtags,
            "target_demographics": self.target_demographics,
            "campaign_id": self.campaign_id,
            "ad_group_id": self.ad_group_id,
            "budget": self.budget,
            "bid_amount": self.bid_amount,
            "daily_budget": self.daily_budget,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "views": self.views,
            "conversions": self.conversions,
            "cost": self.cost,
            "revenue": self.revenue,
            "ctr": self.click_through_rate,
            "cpm": self.cost_per_mille,
            "cpc": self.cost_per_click,
            "cpa": self.cost_per_acquisition,
            "status": self.status,
            "is_approved": self.is_approved,
            "is_paused": self.is_paused,
            "is_active": self.is_active,
            "advertiser_id": str(self.advertiser_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }
