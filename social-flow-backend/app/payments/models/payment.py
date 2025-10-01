"""
Payment model and related schemas.

This module defines the Payment model and related Pydantic schemas
for payment management in the Social Flow backend.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class PaymentStatus(str, Enum):
    """Payment status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class PaymentType(str, Enum):
    """Payment type."""
    SUBSCRIPTION = "subscription"
    ONE_TIME = "one_time"
    CREATOR_PAYOUT = "creator_payout"
    AD_REVENUE = "ad_revenue"
    DONATION = "donation"
    REFUND = "refund"


class Payment(Base):
    """Payment model for storing payment information."""
    
    __tablename__ = "payments"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Payment information
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    payment_type = Column(String(50), nullable=False)
    status = Column(String(20), default=PaymentStatus.PENDING, nullable=False)
    
    # External payment provider information
    provider = Column(String(50), nullable=False)  # stripe, paypal, etc.
    provider_payment_id = Column(String(255), nullable=True)
    provider_transaction_id = Column(String(255), nullable=True)
    
    # Payment method information
    payment_method_id = Column(String(255), nullable=True)
    payment_method_type = Column(String(50), nullable=True)  # card, bank_account, etc.
    last_four_digits = Column(String(4), nullable=True)
    card_brand = Column(String(50), nullable=True)
    card_exp_month = Column(Integer, nullable=True)
    card_exp_year = Column(Integer, nullable=True)
    
    # Billing information
    billing_email = Column(String(255), nullable=True)
    billing_name = Column(String(255), nullable=True)
    billing_address = Column(Text, nullable=True)  # JSON string of address
    
    # Transaction details
    description = Column(Text, nullable=True)
    stripe_metadata = Column(Text, nullable=True)  # JSON string of metadata
    
    # Refund information
    refund_amount = Column(Float, default=0.0, nullable=False)
    refund_reason = Column(Text, nullable=True)
    refunded_at = Column(DateTime, nullable=True)
    
    # Fee information
    processing_fee = Column(Float, default=0.0, nullable=False)
    platform_fee = Column(Float, default=0.0, nullable=False)
    net_amount = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    subscription_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Relationships
    user = relationship("User", back_populates="payments")
    subscription = relationship("Subscription", back_populates="payments")
    
    def __repr__(self) -> str:
        return f"<Payment(id={self.id}, amount={self.amount}, currency={self.currency}, status={self.status})>"
    
    @property
    def is_successful(self) -> bool:
        """Check if payment was successful."""
        return self.status == PaymentStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if payment failed."""
        return self.status == PaymentStatus.FAILED
    
    @property
    def is_refunded(self) -> bool:
        """Check if payment was refunded."""
        return self.status in [PaymentStatus.REFUNDED, PaymentStatus.PARTIALLY_REFUNDED]
    
    @property
    def refund_percentage(self) -> float:
        """Calculate refund percentage."""
        if self.amount == 0:
            return 0.0
        
        return (self.refund_amount / self.amount) * 100
    
    def to_dict(self) -> dict:
        """Convert payment to dictionary."""
        return {
            "id": str(self.id),
            "amount": self.amount,
            "currency": self.currency,
            "payment_type": self.payment_type,
            "status": self.status,
            "provider": self.provider,
            "provider_payment_id": self.provider_payment_id,
            "provider_transaction_id": self.provider_transaction_id,
            "payment_method_id": self.payment_method_id,
            "payment_method_type": self.payment_method_type,
            "last_four_digits": self.last_four_digits,
            "card_brand": self.card_brand,
            "card_exp_month": self.card_exp_month,
            "card_exp_year": self.card_exp_year,
            "billing_email": self.billing_email,
            "billing_name": self.billing_name,
            "billing_address": self.billing_address,
            "description": self.description,
            "metadata": self.stripe_metadata,
            "refund_amount": self.refund_amount,
            "refund_reason": self.refund_reason,
            "refunded_at": self.refunded_at.isoformat() if self.refunded_at else None,
            "processing_fee": self.processing_fee,
            "platform_fee": self.platform_fee,
            "net_amount": self.net_amount,
            "is_successful": self.is_successful,
            "is_failed": self.is_failed,
            "is_refunded": self.is_refunded,
            "refund_percentage": self.refund_percentage,
            "user_id": str(self.user_id),
            "subscription_id": str(self.subscription_id) if self.subscription_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
        }
