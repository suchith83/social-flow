"""
Stripe Connect models for creator payouts.

This module defines models for Stripe Connect integration and creator payouts.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class StripeConnectAccount(Base):
    """Stripe Connect account model for creator payouts."""
    
    __tablename__ = "stripe_connect_accounts"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Stripe details
    stripe_account_id = Column(String(255), unique=True, nullable=False, index=True)
    stripe_customer_id = Column(String(255), nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_onboarded = Column(Boolean, default=False, nullable=False)
    
    # Account details
    account_type = Column(String(50), nullable=True)  # standard, express, custom
    business_type = Column(String(50), nullable=True)  # individual, company
    country = Column(String(2), nullable=True)
    currency = Column(String(3), default='USD', nullable=False)
    
    # Verification
    verification_status = Column(String(50), nullable=True)
    verification_fields_needed = Column(Text, nullable=True)  # JSON array
    
    # Capabilities
    charges_enabled = Column(Boolean, default=False, nullable=False)
    payouts_enabled = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    verified_at = Column(DateTime, nullable=True)
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        unique=True,
        index=True
    )
    
    # Relationships
    user = relationship("User", backref="stripe_connect_account", uselist=False)
    payouts = relationship("CreatorPayout", back_populates="stripe_account", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<StripeConnectAccount(id={self.id}, user_id={self.user_id}, stripe_account_id={self.stripe_account_id})>"


class CreatorPayout(Base):
    """Creator payout model for tracking payments to creators."""
    
    __tablename__ = "creator_payouts"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Payment details
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default='USD', nullable=False)
    description = Column(Text, nullable=True)
    
    # Stripe details
    stripe_payout_id = Column(String(255), unique=True, nullable=True, index=True)
    stripe_transfer_id = Column(String(255), unique=True, nullable=True)
    
    # Status
    status = Column(
        String(50),
        default='pending',
        nullable=False
    )  # pending, processing, paid, failed, cancelled
    
    # Payment breakdown
    watch_time_earnings = Column(Numeric(10, 2), default=Decimal('0'), nullable=False)
    ad_revenue = Column(Numeric(10, 2), default=Decimal('0'), nullable=False)
    subscription_revenue = Column(Numeric(10, 2), default=Decimal('0'), nullable=False)
    donation_revenue = Column(Numeric(10, 2), default=Decimal('0'), nullable=False)
    
    # Period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    paid_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    failure_reason = Column(Text, nullable=True)
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    stripe_account_id = Column(
        UUID(as_uuid=True),
        ForeignKey('stripe_connect_accounts.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    # Relationships
    user = relationship("User", backref="creator_payouts")
    stripe_account = relationship("StripeConnectAccount", back_populates="payouts")
    
    def __repr__(self) -> str:
        return f"<CreatorPayout(id={self.id}, user_id={self.user_id}, amount={self.amount}, status={self.status})>"
