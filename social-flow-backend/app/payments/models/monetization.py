"""
Extended Monetization Models

Additional models for comprehensive monetization system:
- Subscriptions: Platform subscription tiers  
- Ad Campaigns: Advertisement campaigns
- Ad Impressions: Ad view tracking
- Donations: Livestream tips and donations
- Revenue Splits: Automatic revenue sharing
- Payouts: Creator earnings and payouts
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, 
    ForeignKey, Text, Numeric, Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class SubscriptionTier(str, Enum):
    """Subscription tier levels"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    CREATOR = "creator"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    UNPAID = "unpaid"


class PayoutStatus(str, Enum):
    """Payout status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AdCampaignStatus(str, Enum):
    """Ad campaign status"""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    REJECTED = "rejected"


class AdPlacement(str, Enum):
    """Ad placement locations"""
    PRE_ROLL = "pre_roll"
    MID_ROLL = "mid_roll"
    POST_ROLL = "post_roll"
    BANNER = "banner"
    SIDEBAR = "sidebar"
    OVERLAY = "overlay"


# ==================== Subscription Models ====================

class PlatformSubscription(Base):
    """Platform-level subscriptions (distinct from auth Subscription)"""
    __tablename__ = "platform_subscriptions"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Subscription details
    tier = Column(String(20), nullable=False, default=SubscriptionTier.FREE.value)
    status = Column(String(20), nullable=False, default=SubscriptionStatus.ACTIVE.value)
    
    # Stripe integration
    stripe_customer_id = Column(String(100), nullable=True)
    stripe_subscription_id = Column(String(100), nullable=True)
    stripe_price_id = Column(String(100), nullable=True)
    
    # Pricing
    amount = Column(Numeric(10, 2), nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")
    billing_interval = Column(String(20), nullable=False, default="month")  # month, year
    
    # Trial
    trial_start = Column(DateTime(timezone=True), nullable=True)
    trial_end = Column(DateTime(timezone=True), nullable=True)
    
    # Subscription period
    current_period_start = Column(DateTime(timezone=True), nullable=True)
    current_period_end = Column(DateTime(timezone=True), nullable=True)
    
    # Cancellation
    cancel_at_period_end = Column(Boolean, default=False, nullable=False)
    canceled_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Features (JSON for cross-db compat)
    features = Column(JSON, nullable=True, default={})
    
    # Metadata (avoid reserved name 'metadata')
    extra_metadata = Column(JSON, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    payments = relationship("Payment", back_populates="subscription")
    
    # Indexes
    __table_args__ = (
        Index("idx_subscriptions_user", "user_id"),
        Index("idx_subscriptions_status", "status"),
        Index("idx_subscriptions_tier", "tier"),
        Index("idx_subscriptions_stripe_customer", "stripe_customer_id"),
        Index("idx_subscriptions_stripe_subscription", "stripe_subscription_id"),
        Index("idx_subscriptions_period_end", "current_period_end"),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is active"""
        return self.status == SubscriptionStatus.ACTIVE.value
    
    @property
    def is_trial(self) -> bool:
        """Check if in trial period"""
        if not self.trial_end:
            return False
        return datetime.utcnow() < self.trial_end
    
    @property
    def days_until_renewal(self) -> Optional[int]:
        """Days until next billing"""
        if not self.current_period_end:
            return None
        delta = self.current_period_end - datetime.utcnow()
        return max(0, delta.days)


# ==================== Payout Models ====================

class Payout(Base):
    """Creator payouts"""
    __tablename__ = "payouts"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    creator_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Payout details
    status = Column(String(20), nullable=False, default=PayoutStatus.PENDING.value)
    
    # Amount
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    fee = Column(Numeric(10, 2), nullable=False, default=0)
    net_amount = Column(Numeric(10, 2), nullable=False)
    
    # Revenue breakdown
    ad_revenue = Column(Numeric(10, 2), nullable=False, default=0)
    subscription_revenue = Column(Numeric(10, 2), nullable=False, default=0)
    donation_revenue = Column(Numeric(10, 2), nullable=False, default=0)
    other_revenue = Column(Numeric(10, 2), nullable=False, default=0)
    
    # Payment provider
    provider = Column(String(50), nullable=False, default="stripe")
    provider_payout_id = Column(String(100), nullable=True)
    payout_method = Column(String(50), nullable=True)  # bank_account, paypal, etc.
    
    # Period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Bank details (encrypted)
    bank_account_last4 = Column(String(4), nullable=True)
    
    # Failure info
    failure_code = Column(String(50), nullable=True)
    failure_message = Column(Text, nullable=True)
    
    # Metadata (avoid reserved name)
    extra_metadata = Column(JSON, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    creator = relationship("User", back_populates="payouts", foreign_keys=[creator_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_payouts_creator", "creator_id"),
        Index("idx_payouts_status", "status"),
        Index("idx_payouts_period", "period_start", "period_end"),
        Index("idx_payouts_created", "created_at"),
        Index("idx_payouts_provider", "provider_payout_id"),
    )
    
    @property
    def is_completed(self) -> bool:
        """Check if payout completed"""
        return self.status == PayoutStatus.COMPLETED.value


# ==================== Advertisement Models ====================

class AdCampaign(Base):
    """Advertisement campaigns"""
    __tablename__ = "ad_campaigns"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    advertiser_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Campaign details
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default=AdCampaignStatus.DRAFT.value)
    
    # Targeting
    target_audience = Column(JSON, nullable=True, default={})  # age, gender, interests, location
    target_placements = Column(JSON, nullable=True, default=[])  # List of AdPlacement values
    
    # Budget
    daily_budget = Column(Numeric(10, 2), nullable=False)
    total_budget = Column(Numeric(10, 2), nullable=False)
    spent_amount = Column(Numeric(10, 2), nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")
    
    # Pricing model
    pricing_model = Column(String(20), nullable=False, default="cpm")  # cpm, cpc, cpv
    bid_amount = Column(Numeric(10, 4), nullable=False)  # Cost per thousand impressions/clicks/views
    
    # Creative assets
    creative_url = Column(String(500), nullable=False)
    creative_type = Column(String(20), nullable=False)  # image, video
    click_url = Column(String(500), nullable=False)
    
    # Schedule
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=True)
    
    # Performance metrics
    impressions = Column(Integer, nullable=False, default=0)
    clicks = Column(Integer, nullable=False, default=0)
    views = Column(Integer, nullable=False, default=0)
    conversions = Column(Integer, nullable=False, default=0)
    
    # Metadata (avoid reserved name)
    extra_metadata = Column(JSON, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    advertiser = relationship("User", back_populates="ad_campaigns", foreign_keys=[advertiser_id])
    impressions_rel = relationship("AdImpression", back_populates="campaign", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_ad_campaigns_advertiser", "advertiser_id"),
        Index("idx_ad_campaigns_status", "status"),
        Index("idx_ad_campaigns_dates", "start_date", "end_date"),
        CheckConstraint("daily_budget > 0", name="check_daily_budget_positive"),
        CheckConstraint("total_budget > 0", name="check_total_budget_positive"),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if campaign is active"""
        if self.status != AdCampaignStatus.ACTIVE.value:
            return False
        now = datetime.utcnow()
        if now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        if self.spent_amount >= self.total_budget:
            return False
        return True
    
    @property
    def ctr(self) -> float:
        """Click-through rate"""
        if self.impressions == 0:
            return 0.0
        return (self.clicks / self.impressions) * 100
    
    @property
    def budget_remaining(self) -> Decimal:
        """Remaining budget"""
        return self.total_budget - self.spent_amount


class AdImpression(Base):
    """Ad impression tracking"""
    __tablename__ = "ad_impressions"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    campaign_id = Column(PostgresUUID(as_uuid=True), ForeignKey("ad_campaigns.id", ondelete="CASCADE"), nullable=False)
    
    # Where ad was shown
    video_id = Column(PostgresUUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=True)
    stream_id = Column(PostgresUUID(as_uuid=True), ForeignKey("live_streams.id", ondelete="CASCADE"), nullable=True)
    placement = Column(String(20), nullable=False)
    
    # Viewer info
    viewer_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    viewer_ip = Column(String(45), nullable=True)
    viewer_country = Column(String(2), nullable=True)
    viewer_device = Column(String(50), nullable=True)
    
    # Interaction
    was_clicked = Column(Boolean, default=False, nullable=False)
    was_viewed = Column(Boolean, default=False, nullable=False)  # For video ads
    view_duration = Column(Integer, nullable=True)  # Seconds
    
    # Cost
    cost = Column(Numeric(10, 4), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    clicked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    campaign = relationship("AdCampaign", back_populates="impressions_rel")
    video = relationship("Video", back_populates="ad_impressions")
    stream = relationship("LiveStream", back_populates="ad_impressions")
    viewer = relationship("User", back_populates="ad_impressions_viewed", foreign_keys=[viewer_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_ad_impressions_campaign", "campaign_id"),
        Index("idx_ad_impressions_video", "video_id"),
        Index("idx_ad_impressions_stream", "stream_id"),
        Index("idx_ad_impressions_viewer", "viewer_id"),
        Index("idx_ad_impressions_created", "created_at"),
        Index("idx_ad_impressions_clicked", "was_clicked"),
    )


# ==================== Donation Models ====================

class Donation(Base):
    """Livestream donations/tips"""
    __tablename__ = "donations"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    donor_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    recipient_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    stream_id = Column(PostgresUUID(as_uuid=True), ForeignKey("live_streams.id", ondelete="CASCADE"), nullable=True)
    
    # Amount
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    platform_fee = Column(Numeric(10, 2), nullable=False)
    net_amount = Column(Numeric(10, 2), nullable=False)
    
    # Message
    message = Column(Text, nullable=True)
    is_anonymous = Column(Boolean, default=False, nullable=False)
    is_highlighted = Column(Boolean, default=False, nullable=False)  # Super chat style
    
    # Payment provider
    provider = Column(String(50), nullable=False, default="stripe")
    provider_transaction_id = Column(String(100), nullable=True)
    
    # Status
    status = Column(String(20), nullable=False, default="pending")
    refunded = Column(Boolean, default=False, nullable=False)
    refunded_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata (avoid reserved name)
    extra_metadata = Column(JSON, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Relationships
    donor = relationship("User", foreign_keys=[donor_id], back_populates="donations_sent")
    recipient = relationship("User", foreign_keys=[recipient_id], back_populates="donations_received")
    stream = relationship("LiveStream", back_populates="donations")
    
    # Indexes
    __table_args__ = (
        Index("idx_donations_donor", "donor_id"),
        Index("idx_donations_recipient", "recipient_id"),
        Index("idx_donations_stream", "stream_id"),
        Index("idx_donations_created", "created_at"),
        Index("idx_donations_status", "status"),
        CheckConstraint("amount > 0", name="check_donation_amount_positive"),
    )


# ==================== Revenue Split Models ====================

class RevenueSplit(Base):
    """Automatic revenue sharing"""
    __tablename__ = "revenue_splits"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Content reference
    content_id = Column(PostgresUUID(as_uuid=True), nullable=False)
    content_type = Column(String(20), nullable=False)  # video, stream
    
    # Parties
    owner_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    collaborator_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Split percentage (0-100)
    owner_percentage = Column(Numeric(5, 2), nullable=False)
    collaborator_percentage = Column(Numeric(5, 2), nullable=False)
    
    # Revenue tracking
    total_revenue = Column(Numeric(10, 2), nullable=False, default=0)
    owner_earnings = Column(Numeric(10, 2), nullable=False, default=0)
    collaborator_earnings = Column(Numeric(10, 2), nullable=False, default=0)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", foreign_keys=[owner_id], back_populates="revenue_splits_owned")
    collaborator = relationship("User", foreign_keys=[collaborator_id], back_populates="revenue_splits_received")
    
    # Indexes
    __table_args__ = (
        Index("idx_revenue_splits_content", "content_id", "content_type"),
        Index("idx_revenue_splits_owner", "owner_id"),
        Index("idx_revenue_splits_collaborator", "collaborator_id"),
        Index("idx_revenue_splits_active", "is_active"),
        CheckConstraint("owner_percentage + collaborator_percentage = 100", name="check_split_total_100"),
        CheckConstraint("owner_percentage >= 0 AND owner_percentage <= 100", name="check_owner_percentage_range"),
        CheckConstraint("collaborator_percentage >= 0 AND collaborator_percentage <= 100", name="check_collaborator_percentage_range"),
    )
