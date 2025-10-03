"""
Ad model for advertising system.

This module defines the Ad model for managing advertisements.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, Numeric
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from app.core.database import Base


class Ad(Base):
    """Ad model for storing advertisement campaigns."""
    
    __tablename__ = "ads"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Campaign details
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    media_url = Column(String(500), nullable=False)
    media_type = Column(String(50), nullable=False)  # image, video
    click_url = Column(String(500), nullable=False)
    
    # Targeting
    target_age_min = Column(Integer, nullable=True)
    target_age_max = Column(Integer, nullable=True)
    target_gender = Column(String(20), nullable=True)  # male, female, all
    target_locations = Column(Text, nullable=True)  # JSON array of locations
    target_interests = Column(Text, nullable=True)  # JSON array of interests
    target_user_types = Column(Text, nullable=True)  # JSON array: free, premium, creator
    
    # Budget and billing
    budget = Column(Numeric(10, 2), nullable=False)
    cost_per_view = Column(Numeric(10, 4), nullable=False, default=Decimal('0.01'))
    cost_per_click = Column(Numeric(10, 4), nullable=False, default=Decimal('0.10'))
    spent_amount = Column(Numeric(10, 2), default=Decimal('0'), nullable=False)
    
    # Performance metrics
    impressions_count = Column(Integer, default=0, nullable=False)
    views_count = Column(Integer, default=0, nullable=False)  # 7+ seconds
    clicks_count = Column(Integer, default=0, nullable=False)
    conversions_count = Column(Integer, default=0, nullable=False)
    
    # Campaign status
    is_active = Column(Boolean, default=True, nullable=False)
    is_approved = Column(Boolean, default=False, nullable=False)
    is_paused = Column(Boolean, default=False, nullable=False)
    is_completed = Column(Boolean, default=False, nullable=False)
    
    # Schedule
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Moderation
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(UUID(as_uuid=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Foreign keys
    advertiser_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    # Relationships
    advertiser = relationship("User", backref="ads")
    
    def __repr__(self) -> str:
        return f"<Ad(id={self.id}, title={self.title}, advertiser_id={self.advertiser_id})>"
    
    @property
    def ctr(self) -> float:
        """Calculate click-through rate."""
        if self.impressions_count == 0:
            return 0.0
        return (self.clicks_count / self.impressions_count) * 100
    
    @property
    def view_rate(self) -> float:
        """Calculate view rate (7+ seconds)."""
        if self.impressions_count == 0:
            return 0.0
        return (self.views_count / self.impressions_count) * 100
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.clicks_count == 0:
            return 0.0
        return (self.conversions_count / self.clicks_count) * 100
    
    @property
    def budget_remaining(self) -> Decimal:
        """Calculate remaining budget."""
        return Decimal(str(self.budget)) - Decimal(str(self.spent_amount))
    
    @property
    def is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.budget_remaining <= 0
