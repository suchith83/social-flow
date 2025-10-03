"""  
Payment models for Stripe integration and monetization.

This module defines comprehensive payment models including:
- Payments (one-time and recurring)
- Subscriptions (premium memberships)
- Payouts (creator earnings)
- Transactions (audit trail)
- Revenue sharing
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Enum as SQLEnum,
    Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
)
from sqlalchemy.orm import Mapped, relationship

from app.models.base import CommonBase
from app.models.types import JSONB, UUID

if TYPE_CHECKING:
    from app.models.user import User


class PaymentStatus(str, PyEnum):
    """Payment processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    DISPUTED = "disputed"


class PaymentType(str, PyEnum):
    """Payment type classification."""
    SUBSCRIPTION = "subscription"
    ONE_TIME = "one_time"
    CREATOR_PAYOUT = "creator_payout"
    AD_REVENUE = "ad_revenue"
    DONATION = "donation"
    TIP = "tip"
    REFUND = "refund"


class PaymentProvider(str, PyEnum):
    """Payment provider."""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"


class SubscriptionStatus(str, PyEnum):
    """Subscription status."""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    UNPAID = "unpaid"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"


class SubscriptionTier(str, PyEnum):
    """Subscription tier levels."""
    BASIC = "basic"
    PREMIUM = "premium"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class PayoutStatus(str, PyEnum):
    """Payout status."""
    PENDING = "pending"
    PROCESSING = "processing"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Payment(CommonBase):
    """
    Payment model for all payment transactions.
    
    Handles:
    - One-time payments
    - Recurring subscription payments
    - Tips and donations
    - Refunds
    - Stripe integration
    """
    
    __tablename__ = "payments"
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User making the payment"
    )
    
    # ==================== Amount ====================
    amount = Column(
        Float,
        nullable=False,
        doc="Payment amount"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="ISO currency code"
    )
    
    # ==================== Type and Status ====================
    payment_type = Column(
        SQLEnum(PaymentType),
        nullable=False,
        index=True,
        doc="Payment type"
    )
    
    status = Column(
        SQLEnum(PaymentStatus),
        default=PaymentStatus.PENDING,
        nullable=False,
        index=True,
        doc="Payment status"
    )
    
    # ==================== Payment Provider ====================
    provider = Column(
        SQLEnum(PaymentProvider),
        default=PaymentProvider.STRIPE,
        nullable=False,
        doc="Payment provider"
    )
    
    provider_payment_id = Column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        doc="Provider payment ID (e.g., Stripe payment intent ID)"
    )
    
    provider_transaction_id = Column(
        String(255),
        nullable=True,
        index=True,
        doc="Provider transaction ID"
    )
    
    # ==================== Payment Method ====================
    payment_method_id = Column(
        String(255),
        nullable=True,
        doc="Stripe payment method ID"
    )
    
    payment_method_type = Column(
        String(50),
        nullable=True,
        doc="Payment method type: card, bank_account, etc."
    )
    
    card_brand = Column(
        String(50),
        nullable=True,
        doc="Card brand: visa, mastercard, amex, etc."
    )
    
    card_last4 = Column(
        String(4),
        nullable=True,
        doc="Last 4 digits of card"
    )
    
    card_exp_month = Column(
        Integer,
        nullable=True,
        doc="Card expiration month"
    )
    
    card_exp_year = Column(
        Integer,
        nullable=True,
        doc="Card expiration year"
    )
    
    # ==================== Billing Information ====================
    billing_email = Column(
        String(255),
        nullable=True,
        doc="Billing email address"
    )
    
    billing_name = Column(
        String(255),
        nullable=True,
        doc="Billing name"
    )
    
    billing_address = Column(
        JSONB,
        nullable=True,
        doc="Billing address (structured JSON)"
    )
    
    # ==================== Transaction Details ====================
    description = Column(
        Text,
        nullable=True,
        doc="Payment description"
    )
    
    # ==================== Fees ====================
    processing_fee = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Payment processing fee (Stripe, etc.)"
    )
    
    platform_fee = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Platform commission fee"
    )
    
    net_amount = Column(
        Float,
        nullable=False,
        doc="Net amount after fees"
    )
    
    # ==================== Refund Information ====================
    refunded_amount = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Total refunded amount"
    )
    
    refund_reason = Column(
        Text,
        nullable=True,
        doc="Refund reason"
    )
    
    refunded_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Refund timestamp"
    )
    
    # ==================== Timestamps ====================
    processed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Payment processing completion timestamp"
    )
    
    failed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Payment failure timestamp"
    )
    
    # ==================== Related Records ====================
    subscription_id = Column(
        UUID(as_uuid=True),
        ForeignKey('subscriptions.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Related subscription ID"
    )
    
    payout_id = Column(
        UUID(as_uuid=True),
        ForeignKey('payouts.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Related payout ID (for creator payments)"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        back_populates="payments",
        foreign_keys=[user_id]
    )
    
    subscription: Mapped["Subscription"] = relationship("Subscription", back_populates="payments")
    payout: Mapped["Payout"] = relationship("Payout", back_populates="payments")
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_payment_user_status', 'user_id', 'status'),
        Index('idx_payment_type_status', 'payment_type', 'status'),
        Index('idx_payment_provider_id', 'provider', 'provider_payment_id'),
        Index('idx_payment_created', 'created_at'),
        Index('idx_payment_processed', 'processed_at'),
        
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<Payment(id={self.id}, amount={self.amount}, status={self.status})>"


class Subscription(CommonBase):
    """
    Subscription model for recurring payments.
    
    Handles premium memberships and recurring subscriptions.
    """
    
    __tablename__ = "subscriptions"
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Subscriber user ID"
    )
    
    # ==================== Subscription Details ====================
    tier = Column(
        SQLEnum(SubscriptionTier),
        nullable=False,
        index=True,
        doc="Subscription tier"
    )
    
    status = Column(
        SQLEnum(SubscriptionStatus),
        default=SubscriptionStatus.INCOMPLETE,
        nullable=False,
        index=True,
        doc="Subscription status"
    )
    
    # ==================== Stripe ====================
    stripe_subscription_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Stripe subscription ID"
    )
    
    stripe_customer_id = Column(
        String(255),
        nullable=True,
        index=True,
        doc="Stripe customer ID"
    )
    
    stripe_price_id = Column(
        String(255),
        nullable=True,
        doc="Stripe price ID"
    )
    
    # ==================== Pricing ====================
    price_amount = Column(
        Float,
        nullable=False,
        doc="Subscription price per billing cycle"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Currency code"
    )
    
    billing_cycle = Column(
        String(20),
        default="monthly",
        nullable=False,
        doc="Billing cycle: monthly, yearly, etc."
    )
    
    # ==================== Trial ====================
    trial_ends_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Trial period end timestamp"
    )
    
    trial_days = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of trial days"
    )
    
    # ==================== Billing Dates ====================
    current_period_start = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Current billing period start"
    )
    
    current_period_end = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Current billing period end"
    )
    
    # ==================== Cancellation ====================
    cancel_at_period_end = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether to cancel at period end"
    )
    
    cancelled_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Cancellation timestamp"
    )
    
    cancellation_reason = Column(
        Text,
        nullable=True,
        doc="Cancellation reason"
    )
    
    # ==================== Timestamps ====================
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Subscription start timestamp"
    )
    
    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Subscription end timestamp"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        back_populates="subscriptions",
        foreign_keys=[user_id]
    )
    
    payments: Mapped[list["Payment"]] = relationship(
        "Payment",
        back_populates="subscription",
        cascade="all, delete-orphan"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_subscription_user_status', 'user_id', 'status'),
        Index('idx_subscription_tier_status', 'tier', 'status'),
        Index('idx_subscription_period_end', 'current_period_end'),
        Index('idx_subscription_stripe', 'stripe_subscription_id'),
    )
    
    def __repr__(self) -> str:
        return f"<Subscription(id={self.id}, user_id={self.user_id}, tier={self.tier})>"
    
    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]


class Payout(CommonBase):
    """
    Payout model for creator earnings.
    
    Tracks payments from platform to creators through Stripe Connect.
    """
    
    __tablename__ = "payouts"
    
    # ==================== Creator ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="Creator user ID receiving payout"
    )
    
    # ==================== Amount ====================
    amount = Column(
        Float,
        nullable=False,
        doc="Payout amount"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Currency code"
    )
    
    # ==================== Status ====================
    status = Column(
        SQLEnum(PayoutStatus),
        default=PayoutStatus.PENDING,
        nullable=False,
        index=True,
        doc="Payout status"
    )
    
    # ==================== Stripe Connect ====================
    stripe_payout_id = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Stripe payout ID"
    )
    
    stripe_connect_account_id = Column(
        String(255),
        nullable=True,
        doc="Stripe Connect account ID"
    )
    
    # ==================== Period ====================
    period_start = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Earnings period start"
    )
    
    period_end = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Earnings period end"
    )
    
    # ==================== Breakdown ====================
    ad_revenue = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Revenue from ads"
    )
    
    subscription_revenue = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Revenue from subscriptions"
    )
    
    tip_revenue = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Revenue from tips/donations"
    )
    
    other_revenue = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Other revenue sources"
    )
    
    # ==================== Fees ====================
    platform_fee = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Platform commission"
    )
    
    processing_fee = Column(
        Float,
        default=0.0,
        nullable=False,
        doc="Payment processing fee"
    )
    
    net_amount = Column(
        Float,
        nullable=False,
        doc="Net payout amount after fees"
    )
    
    # ==================== Bank Details ====================
    bank_name = Column(
        String(100),
        nullable=True,
        doc="Bank name"
    )
    
    account_last4 = Column(
        String(4),
        nullable=True,
        doc="Last 4 digits of bank account"
    )
    
    # ==================== Timestamps ====================
    paid_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Payout completion timestamp"
    )
    
    failed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Payout failure timestamp"
    )
    
    # ==================== Failure ====================
    failure_reason = Column(
        Text,
        nullable=True,
        doc="Failure reason if payout failed"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship(
        "User",
        backref="payouts",
        foreign_keys=[user_id]
    )
    
    payments: Mapped[list["Payment"]] = relationship(
        "Payment",
        back_populates="payout"
    )
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_payout_user_status', 'user_id', 'status'),
        Index('idx_payout_period', 'period_start', 'period_end'),
        Index('idx_payout_created', 'created_at'),
        Index('idx_payout_paid', 'paid_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Payout(id={self.id}, user_id={self.user_id}, amount={self.amount})>"


class Transaction(CommonBase):
    """
    Transaction audit trail model.
    
    Immutable ledger of all financial transactions for audit purposes.
    """
    
    __tablename__ = "transactions"
    
    # ==================== User ====================
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc="User involved in transaction"
    )
    
    # ==================== Transaction Details ====================
    transaction_type = Column(
        String(50),
        nullable=False,
        index=True,
        doc="Transaction type: charge, refund, payout, etc."
    )
    
    amount = Column(
        Float,
        nullable=False,
        doc="Transaction amount"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Currency code"
    )
    
    balance_before = Column(
        Float,
        nullable=False,
        doc="User balance before transaction"
    )
    
    balance_after = Column(
        Float,
        nullable=False,
        doc="User balance after transaction"
    )
    
    # ==================== Description ====================
    description = Column(
        Text,
        nullable=False,
        doc="Transaction description"
    )
    
    # ==================== Related Records ====================
    payment_id = Column(
        UUID(as_uuid=True),
        ForeignKey('payments.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Related payment ID"
    )
    
    payout_id = Column(
        UUID(as_uuid=True),
        ForeignKey('payouts.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Related payout ID"
    )
    
    # ==================== Relationships ====================
    user: Mapped["User"] = relationship("User", backref="transactions")
    payment: Mapped["Payment"] = relationship("Payment", backref="transactions")
    payout: Mapped["Payout"] = relationship("Payout", backref="transactions")
    
    # ==================== Table Configuration ====================
    __table_args__ = (
        Index('idx_transaction_user_created', 'user_id', 'created_at'),
        Index('idx_transaction_type_created', 'transaction_type', 'created_at'),
        Index('idx_transaction_payment', 'payment_id'),
        Index('idx_transaction_payout', 'payout_id'),
        
        # Partition by created_at (monthly) for efficient querying
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
    
    def __repr__(self) -> str:
        return f"<Transaction(id={self.id}, type={self.transaction_type}, amount={self.amount})>"


# Export models
__all__ = [
    'Payment',
    'Subscription',
    'Payout',
    'Transaction',
    'PaymentStatus',
    'PaymentType',
    'PaymentProvider',
    'SubscriptionStatus',
    'SubscriptionTier',
    'PayoutStatus',
]
