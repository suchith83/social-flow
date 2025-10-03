"""
Payment and Subscription Schemas.

Pydantic schemas for payment, subscription, and payout operations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field
from pydantic import model_validator


# ============================================================================
# PAYMENT SCHEMAS
# ============================================================================

class PaymentIntentCreate(BaseModel):
    """Create payment intent request."""
    amount: float = Field(..., gt=0, description="Payment amount")
    currency: str = Field(default="usd", max_length=3)
    payment_type: Optional[str] = Field("one_time", description="Payment type (one_time, donation, etc.)")
    description: Optional[str] = Field(None, max_length=500)
    metadata: Optional[Dict[str, Any]] = None
    # Allow extra test field commonly provided in unit tests
    payment_method_id: Optional[str] = None


class PaymentIntentResponse(BaseModel):
    """Payment intent response."""
    payment_id: str
    client_secret: str
    stripe_payment_intent_id: str
    amount: float
    currency: str
    status: str


class PaymentConfirm(BaseModel):
    """Confirm payment request."""
    payment_id: str


class PaymentRefund(BaseModel):
    """Refund payment request."""
    payment_id: Optional[str] = None  # Optional since payment_id is in URL path
    amount: Optional[float] = Field(None, gt=0, description="Partial refund amount")
    reason: Optional[str] = Field(None, max_length=500)


class PaymentResponse(BaseModel):
    """Payment response."""
    id: str
    amount: float
    currency: str
    payment_type: str
    status: str
    provider: str
    provider_payment_id: Optional[str]
    payment_method_type: Optional[str]
    last_four_digits: Optional[str]
    card_brand: Optional[str]
    description: Optional[str]
    refund_amount: float
    processing_fee: float
    platform_fee: float
    net_amount: float
    created_at: datetime
    processed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PaymentList(BaseModel):
    """Payment list response."""
    payments: List[PaymentResponse]
    total_count: int
    page: int
    page_size: int


# ============================================================================
# SUBSCRIPTION SCHEMAS
# ============================================================================

class SubscriptionTierEnum(str, Enum):
    """Subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionCreate(BaseModel):
    """Create subscription request."""
    tier: SubscriptionTierEnum
    payment_method_id: str = Field(..., description="Stripe payment method ID")
    trial_days: int = Field(default=0, ge=0, le=30)

    @model_validator(mode="before")
    @classmethod
    def _accept_plan_alias(cls, values: Dict[str, Any]):
        """Accept 'plan' as an alias for 'tier' to match older tests."""
        if isinstance(values, dict):
            if "tier" not in values and "plan" in values:
                values = {**values, "tier": values.get("plan")}
        return values

    @property
    def plan(self) -> str:
        """Back-compat accessor used by some tests."""
        try:
            return self.tier.value  # enum value like 'premium'
        except Exception:
            return str(self.tier)


class SubscriptionUpdate(BaseModel):
    """Update subscription request."""
    subscription_id: str
    new_tier: SubscriptionTierEnum


class SubscriptionCancel(BaseModel):
    """Cancel subscription request."""
    subscription_id: str
    immediate: bool = Field(default=False, description="Cancel immediately vs at period end")


class SubscriptionResponse(BaseModel):
    """Subscription response."""
    id: str
    tier: str
    status: str
    price: float
    currency: str
    billing_cycle: str
    is_trial: bool
    trial_days: int
    trial_days_remaining: int
    days_remaining: int
    is_active: bool
    is_trial_active: bool
    start_date: datetime
    end_date: Optional[datetime]
    trial_start_date: Optional[datetime]
    trial_end_date: Optional[datetime]
    cancelled_at: Optional[datetime]
    features: Optional[str]
    limits: Optional[str]
    provider_subscription_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SubscriptionPricing(BaseModel):
    """Subscription pricing information."""
    tier: str
    display_name: str
    price: float
    currency: str
    billing_cycle: str
    features: List[str]
    limits: Dict[str, int]
    is_popular: bool = False


class SubscriptionPricingList(BaseModel):
    """List of subscription pricing options."""
    pricing: List[SubscriptionPricing]


# ============================================================================
# STRIPE CONNECT & PAYOUT SCHEMAS
# ============================================================================

class ConnectAccountCreate(BaseModel):
    """Create Connect account request."""
    country: str = Field(default="US", max_length=2)
    account_type: str = Field(default="express", description="express, standard, or custom")


class ConnectAccountResponse(BaseModel):
    """Connect account response."""
    connect_account_id: str
    stripe_account_id: str
    onboarding_url: Optional[str]
    status: str
    charges_enabled: bool
    payouts_enabled: bool
    details_submitted: bool
    is_fully_onboarded: bool
    requirements_currently_due: List[str] = []
    available_balance: float
    pending_balance: float
    currency: str
    total_volume: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConnectAccountStatus(BaseModel):
    """Connect account status."""
    connect_account_id: str
    stripe_account_id: str
    status: str
    charges_enabled: bool
    payouts_enabled: bool
    details_submitted: bool
    is_fully_onboarded: bool
    requirements_currently_due: List[str]
    available_balance: float
    pending_balance: float


class PayoutCreate(BaseModel):
    """Create payout request."""
    period_start: datetime
    period_end: datetime
    revenue_breakdown: Dict[str, float] = Field(
        ...,
        description="Revenue breakdown by source (subscription, tips, content_sales, ad_revenue)"
    )


class PayoutResponse(BaseModel):
    """Payout response."""
    id: str
    amount: float
    currency: str
    status: str
    stripe_payout_id: Optional[str]
    stripe_transfer_id: Optional[str]
    gross_amount: float
    platform_fee: float
    stripe_fee: float
    net_amount: float
    subscription_revenue: float
    tips_revenue: float
    content_sales_revenue: float
    ad_revenue: float
    period_start: datetime
    period_end: datetime
    description: Optional[str]
    failure_message: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]
    paid_at: Optional[datetime]
    failed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PayoutList(BaseModel):
    """Payout list response."""
    payouts: List[PayoutResponse]
    total_count: int
    total_amount: float
    page: int
    page_size: int


class RevenueBreakdown(BaseModel):
    """Revenue breakdown for payout period."""
    period_start: datetime
    period_end: datetime
    subscription_revenue: float
    tips_revenue: float
    content_sales_revenue: float
    ad_revenue: float
    total_revenue: float
    platform_fee: float
    stripe_fee: float
    net_revenue: float


# ============================================================================
# PAYMENT METHOD SCHEMAS
# ============================================================================

class PaymentMethodAttach(BaseModel):
    """Attach payment method request."""
    payment_method_id: str = Field(..., description="Stripe payment method ID")


class PaymentMethodDetach(BaseModel):
    """Detach payment method request."""
    payment_method_id: str


class PaymentMethodResponse(BaseModel):
    """Payment method response."""
    id: str
    type: str
    card_brand: Optional[str]
    card_last_four: Optional[str]
    card_exp_month: Optional[int]
    card_exp_year: Optional[int]
    is_default: bool
    created_at: datetime


class PaymentMethodList(BaseModel):
    """Payment method list response."""
    payment_methods: List[PaymentMethodResponse]


# ============================================================================
# WEBHOOK SCHEMAS
# ============================================================================

class WebhookEventResponse(BaseModel):
    """Webhook event response."""
    id: str
    stripe_event_id: str
    event_type: str
    is_processed: bool
    processing_error: Optional[str]
    retry_count: int
    created_at: datetime
    processed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


# ============================================================================
# ANALYTICS SCHEMAS
# ============================================================================

class PaymentAnalytics(BaseModel):
    """Payment analytics response."""
    total_revenue: float
    total_transactions: int
    successful_transactions: int
    failed_transactions: int
    refunded_transactions: int
    average_transaction_value: float
    total_platform_fees: float
    total_stripe_fees: float
    net_revenue: float
    period_start: datetime
    period_end: datetime


class SubscriptionAnalytics(BaseModel):
    """Subscription analytics response."""
    active_subscriptions: int
    trial_subscriptions: int
    cancelled_subscriptions: int
    monthly_recurring_revenue: float
    annual_recurring_revenue: float
    average_subscription_value: float
    churn_rate: float
    retention_rate: float
    period_start: datetime
    period_end: datetime


class CreatorEarnings(BaseModel):
    """Creator earnings summary."""
    user_id: str
    total_earnings: float
    pending_earnings: float
    paid_earnings: float
    subscription_earnings: float
    tips_earnings: float
    content_sales_earnings: float
    ad_revenue_earnings: float
    platform_fees_paid: float
    stripe_fees_paid: float
    next_payout_date: Optional[datetime]
    next_payout_amount: float
    currency: str


# Backwards compatibility aliases for tests
PaymentCreate = PaymentIntentCreate
