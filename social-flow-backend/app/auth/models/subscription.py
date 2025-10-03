"""
Subscription model and related schemas.

This module defines the Subscription model and related Pydantic schemas
for subscription management in the Social Flow backend.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class SubscriptionStatus(str, Enum):
    """Subscription status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    EXPIRED = "expired"
    TRIAL = "trial"


class SubscriptionTier(str, Enum):
    """Subscription tier."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class Subscription(Base):
    """Subscription model for storing subscription information."""
    
    __tablename__ = "subscriptions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Subscription information
    tier = Column(String(50), nullable=False)
    status = Column(String(20), default=SubscriptionStatus.ACTIVE, nullable=False)
    
    # Pricing information
    price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    billing_cycle = Column(String(20), nullable=False)  # monthly, yearly, lifetime
    
    # Trial information
    is_trial = Column(Boolean, default=False, nullable=False)
    trial_days = Column(Integer, default=0, nullable=False)
    trial_used = Column(Boolean, default=False, nullable=False)
    
    # Billing information
    billing_email = Column(String(255), nullable=True)
    billing_name = Column(String(255), nullable=True)
    billing_address = Column(Text, nullable=True)  # JSON string of address
    
    # Payment method information
    payment_method_id = Column(String(255), nullable=True)
    payment_method_type = Column(String(50), nullable=True)
    
    # External provider information
    provider = Column(String(50), nullable=False)  # stripe, paypal, etc.
    provider_subscription_id = Column(String(255), nullable=True)
    provider_customer_id = Column(String(255), nullable=True)
    
    # Subscription dates
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    trial_start_date = Column(DateTime, nullable=True)
    trial_end_date = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Features and limits
    features = Column(Text, nullable=True)  # JSON string of features
    limits = Column(Text, nullable=True)  # JSON string of limits
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    payments = relationship("Payment", back_populates="subscription", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Subscription(id={self.id}, tier={self.tier}, status={self.status}, user_id={self.user_id})>"

    def __init__(self, *args, **kwargs):
        """Allow test-friendly alias fields used in unit tests.
        - plan -> tier
        - stripe_subscription_id -> provider_subscription_id
        - amount -> price
        - current_period_start -> start_date
        - current_period_end -> end_date
        - currency, billing_cycle passed through
        """
        alias_map = {
            "plan": "tier",
            "stripe_subscription_id": "provider_subscription_id",
            "amount": "price",
            "current_period_start": "start_date",
            "current_period_end": "end_date",
        }
        for alias, real in alias_map.items():
            if alias in kwargs and real not in kwargs:
                kwargs[real] = kwargs.pop(alias)
        super().__init__(*args, **kwargs)
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        if self.status != SubscriptionStatus.ACTIVE:
            return False
        
        if self.end_date and datetime.utcnow() > self.end_date:
            return False
        
        return True
    
    @property
    def is_trial_active(self) -> bool:
        """Check if trial is active."""
        if not self.is_trial or self.trial_used:
            return False
        
        if not self.trial_start_date or not self.trial_end_date:
            return False
        
        now = datetime.utcnow()
        return self.trial_start_date <= now <= self.trial_end_date
    
    @property
    def days_remaining(self) -> int:
        """Calculate days remaining in subscription."""
        if not self.end_date:
            return -1  # No end date (lifetime)
        
        remaining = (self.end_date - datetime.utcnow()).days
        return max(0, remaining)
    
    @property
    def trial_days_remaining(self) -> int:
        """Calculate trial days remaining."""
        if not self.is_trial_active:
            return 0
        
        remaining = (self.trial_end_date - datetime.utcnow()).days
        return max(0, remaining)
    
    def to_dict(self) -> dict:
        """Convert subscription to dictionary."""
        return {
            "id": str(self.id),
            "tier": self.tier,
            "status": self.status,
            "price": self.price,
            "currency": self.currency,
            "billing_cycle": self.billing_cycle,
            "is_trial": self.is_trial,
            "trial_days": self.trial_days,
            "trial_used": self.trial_used,
            "billing_email": self.billing_email,
            "billing_name": self.billing_name,
            "billing_address": self.billing_address,
            "payment_method_id": self.payment_method_id,
            "payment_method_type": self.payment_method_type,
            "provider": self.provider,
            "provider_subscription_id": self.provider_subscription_id,
            "provider_customer_id": self.provider_customer_id,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "trial_start_date": self.trial_start_date.isoformat() if self.trial_start_date else None,
            "trial_end_date": self.trial_end_date.isoformat() if self.trial_end_date else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "features": self.features,
            "limits": self.limits,
            "is_active": self.is_active,
            "is_trial_active": self.is_trial_active,
            "days_remaining": self.days_remaining,
            "trial_days_remaining": self.trial_days_remaining,
            "user_id": str(self.user_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
