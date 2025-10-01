"""
Stripe Connect and Payout models.

Models for managing creator payouts through Stripe Connect.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ConnectAccountStatus(str, Enum):
    """Stripe Connect account status."""
    PENDING = "pending"
    ACTIVE = "active"
    RESTRICTED = "restricted"
    DISABLED = "disabled"


class PayoutStatus(str, Enum):
    """Payout status."""
    PENDING = "pending"
    PROCESSING = "processing"
    IN_TRANSIT = "in_transit"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StripeConnectAccount(Base):
    """Stripe Connect account for creator payouts."""
    
    __tablename__ = "stripe_connect_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)
    
    # Stripe account details
    stripe_account_id = Column(String(255), unique=True, nullable=False, index=True)
    account_type = Column(String(50), nullable=False)  # express, standard, custom
    country = Column(String(2), nullable=False)
    
    # Account status
    status = Column(String(20), default=ConnectAccountStatus.PENDING, nullable=False)
    charges_enabled = Column(Boolean, default=False, nullable=False)
    payouts_enabled = Column(Boolean, default=False, nullable=False)
    details_submitted = Column(Boolean, default=False, nullable=False)
    
    # Requirements
    requirements_currently_due = Column(Text, nullable=True)  # JSON array
    requirements_eventually_due = Column(Text, nullable=True)  # JSON array
    requirements_past_due = Column(Text, nullable=True)  # JSON array
    
    # Account details
    business_type = Column(String(50), nullable=True)  # individual, company
    business_name = Column(String(255), nullable=True)
    business_url = Column(String(500), nullable=True)
    support_email = Column(String(255), nullable=True)
    support_phone = Column(String(50), nullable=True)
    
    # Bank account
    external_account_id = Column(String(255), nullable=True)
    bank_name = Column(String(255), nullable=True)
    bank_last_four = Column(String(4), nullable=True)
    
    # Revenue tracking
    total_volume = Column(Float, default=0.0, nullable=False)
    available_balance = Column(Float, default=0.0, nullable=False)
    pending_balance = Column(Float, default=0.0, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    
    # Onboarding
    onboarding_url = Column(Text, nullable=True)
    onboarding_completed_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    payouts = relationship("CreatorPayout", back_populates="connect_account", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<StripeConnectAccount(id={self.id}, stripe_account_id={self.stripe_account_id}, status={self.status})>"
    
    @property
    def is_fully_onboarded(self) -> bool:
        """Check if account is fully onboarded."""
        return (
            self.details_submitted
            and self.charges_enabled
            and self.payouts_enabled
            and self.status == ConnectAccountStatus.ACTIVE
        )


class CreatorPayout(Base):
    """Creator payout tracking."""
    
    __tablename__ = "creator_payouts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    connect_account_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Payout details
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    status = Column(String(20), default=PayoutStatus.PENDING, nullable=False)
    
    # Stripe payout details
    stripe_payout_id = Column(String(255), unique=True, nullable=True, index=True)
    stripe_transfer_id = Column(String(255), unique=True, nullable=True)
    
    # Payout breakdown
    gross_amount = Column(Float, nullable=False)
    platform_fee = Column(Float, default=0.0, nullable=False)
    stripe_fee = Column(Float, default=0.0, nullable=False)
    net_amount = Column(Float, nullable=False)
    
    # Revenue source breakdown
    subscription_revenue = Column(Float, default=0.0, nullable=False)
    tips_revenue = Column(Float, default=0.0, nullable=False)
    content_sales_revenue = Column(Float, default=0.0, nullable=False)
    ad_revenue = Column(Float, default=0.0, nullable=False)
    
    # Payout period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Description and notes
    description = Column(Text, nullable=True)
    stripe_metadata = Column(Text, nullable=True)  # JSON
    failure_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    scheduled_at = Column(DateTime, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    
    # Relationships
    connect_account = relationship("StripeConnectAccount", back_populates="payouts")
    
    def __repr__(self) -> str:
        return f"<CreatorPayout(id={self.id}, amount={self.amount}, status={self.status})>"


class WebhookEvent(Base):
    """Stripe webhook event tracking."""
    
    __tablename__ = "stripe_webhook_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Stripe event details
    stripe_event_id = Column(String(255), unique=True, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    event_version = Column(String(50), nullable=True)
    
    # Event data
    event_data = Column(Text, nullable=False)  # Full JSON payload
    
    # Processing status
    is_processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"<WebhookEvent(id={self.id}, event_type={self.event_type}, is_processed={self.is_processed})>"
