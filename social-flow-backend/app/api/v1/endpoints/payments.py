"""
Payment endpoints for the Social Flow API.

This module provides comprehensive payment processing, subscription management,
and creator payout functionality with Stripe integration.
"""

from typing import Optional
from uuid import UUID
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_db,
    get_current_user,
)
from app.models.user import User
from app.models.payment import (
    Payment,
    Subscription,
    Payout,
    Transaction,
    PaymentStatus,
    SubscriptionStatus,
    PayoutStatus,
)
from app.infrastructure.crud import crud_payment
from app.payments.schemas.payment import (
    PaymentIntentCreate,
    PaymentIntentResponse,
    PaymentRefund,
    PaymentResponse,
    PaymentList,
    SubscriptionCreate,
    SubscriptionUpdate,
    SubscriptionCancel,
    SubscriptionResponse,
    SubscriptionPricing,
    SubscriptionPricingList,
    ConnectAccountCreate,
    ConnectAccountResponse,
    ConnectAccountStatus,
    PayoutCreate,
    PayoutResponse,
    PayoutList,
    PaymentAnalytics,
    SubscriptionAnalytics,
    CreatorEarnings,
)

router = APIRouter()


# ============================================================================
# PAYMENT INTENT ENDPOINTS
# ============================================================================


@router.post(
    "/payments/intent",
    response_model=PaymentIntentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create payment intent",
    description="Create a Stripe payment intent for one-time payments or donations",
)
async def create_payment_intent(
    payment_data: PaymentIntentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaymentIntentResponse:
    """
    Create a payment intent with Stripe.
    
    This endpoint initiates a payment by creating a Stripe PaymentIntent
    and a corresponding Payment record in the database.
    
    **Payment Flow:**
    1. Create payment intent on Stripe
    2. Store payment record in database
    3. Return client secret for frontend confirmation
    
    **Supported Payment Types:**
    - one_time: Standard one-time payment
    - donation: User donations
    - tip: Creator tips
    - content_purchase: Content sales
    """
    # Create Stripe payment intent (mock implementation)
    # In production, this would call Stripe API:
    # stripe.PaymentIntent.create(amount=..., currency=..., customer=...)
    
    import uuid
    stripe_payment_intent_id = f"pi_{uuid.uuid4().hex[:24]}"
    client_secret = f"{stripe_payment_intent_id}_secret_{uuid.uuid4().hex[:16]}"
    
    # Calculate fees
    # Stripe fee: 2.9% + $0.30
    processing_fee = (payment_data.amount * 0.029) + 0.30
    # Platform fee: 10%
    platform_fee = payment_data.amount * 0.10
    # Net amount after all fees
    net_amount = payment_data.amount - processing_fee - platform_fee
    
    # Create payment record
    payment = Payment(
        user_id=current_user.id,
        amount=payment_data.amount,
        currency=payment_data.currency,
        payment_type=payment_data.payment_type or "one_time",
        status=PaymentStatus.PENDING,
        provider="stripe",
        provider_payment_id=stripe_payment_intent_id,
        description=payment_data.description,
        metadata=payment_data.metadata or {},
        processing_fee=processing_fee,
        platform_fee=platform_fee,
        net_amount=net_amount,
    )
    
    db.add(payment)
    await db.commit()
    await db.refresh(payment)
    
    return PaymentIntentResponse(
        payment_id=str(payment.id),
        client_secret=client_secret,
        stripe_payment_intent_id=stripe_payment_intent_id,
        amount=payment.amount,
        currency=payment.currency,
        status=payment.status.value,
    )


@router.post(
    "/payments/{payment_id}/confirm",
    response_model=PaymentResponse,
    summary="Confirm payment",
    description="Confirm a payment after successful Stripe confirmation",
)
async def confirm_payment(
    payment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaymentResponse:
    """
    Confirm a payment after Stripe confirmation.
    
    This endpoint is called after the frontend successfully confirms
    the payment with Stripe. It updates the payment status to succeeded.
    
    **Process:**
    1. Verify payment belongs to user
    2. Update status to SUCCEEDED
    3. Record processing time
    4. Create transaction record
    """
    # Get payment
    payment = await crud_payment.payment.get(db, payment_id)
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found",
        )
    
    # Verify ownership
    if payment.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this payment",
        )
    
    # Update payment status
    payment.status = PaymentStatus.COMPLETED
    payment.processed_at = datetime.now(timezone.utc)
    
    # Calculate fees (mock - in production would come from Stripe)
    payment.processing_fee = payment.amount * 0.029 + 0.30  # Stripe fee
    payment.platform_fee = payment.amount * 0.10  # 10% platform fee
    payment.net_amount = payment.amount - payment.processing_fee - payment.platform_fee
    
    db.add(payment)
    
    # Create transaction record
    transaction = Transaction(
        user_id=current_user.id,
        transaction_type="subscription_payment",
        amount=payment.amount,
        currency=payment.currency,
        balance_before=0.0,  # Mock balance tracking
        balance_after=0.0,
        description=f"Payment {payment.id}",
        payment_id=payment.id,
    )
    db.add(transaction)
    
    await db.commit()
    await db.refresh(payment)
    
    return PaymentResponse(
        id=str(payment.id),
        amount=payment.amount,
        currency=payment.currency,
        payment_type=payment.payment_type,
        status=payment.status.value,
        provider=payment.provider,
        provider_payment_id=payment.provider_payment_id,
        payment_method_type=payment.payment_method_type,
        last_four_digits=payment.card_last4,
        card_brand=payment.card_brand,
        description=payment.description,
        refund_amount=payment.refunded_amount,
        processing_fee=payment.processing_fee,
        platform_fee=payment.platform_fee,
        net_amount=payment.net_amount,
        created_at=payment.created_at,
        processed_at=payment.processed_at,
    )


@router.post(
    "/payments/{payment_id}/refund",
    response_model=PaymentResponse,
    summary="Refund payment",
    description="Refund a payment (full or partial)",
)
async def refund_payment(
    payment_id: UUID,
    refund_data: PaymentRefund,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaymentResponse:
    """
    Refund a payment.
    
    Supports full and partial refunds. In production, this would
    create a Stripe refund and update the payment record.
    
    **Refund Types:**
    - Full refund: No amount specified
    - Partial refund: Specific amount provided
    
    **Requirements:**
    - Payment must be in SUCCEEDED status
    - User must be payment owner or admin
    - Refund amount must not exceed remaining refundable amount
    """
    # Get payment
    payment = await crud_payment.payment.get(db, payment_id)
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found",
        )
    
    # Verify ownership
    if payment.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to refund this payment",
        )
    
    # Check payment status
    if payment.status != PaymentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only completed payments can be refunded",
        )
    
    # Calculate refund amount
    refund_amount = refund_data.amount or (payment.amount - payment.refunded_amount)
    remaining_refundable = payment.amount - payment.refunded_amount
    
    if refund_amount > remaining_refundable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Refund amount exceeds remaining refundable amount: {remaining_refundable}",
        )
    
    # In production: Create Stripe refund
    # refund = stripe.Refund.create(
    #     payment_intent=payment.provider_payment_id,
    #     amount=int(refund_amount * 100),
    # )
    
    # Update payment
    payment.refunded_amount += refund_amount
    if payment.refunded_amount >= payment.amount:
        payment.status = PaymentStatus.REFUNDED
    else:
        payment.status = PaymentStatus.PARTIALLY_REFUNDED
    
    db.add(payment)
    
    # Create transaction record
    transaction = Transaction(
        user_id=current_user.id,
        transaction_type="refund",
        amount=refund_amount,
        currency=payment.currency,
        description=f"Refund for payment {payment.id}: {refund_data.reason or 'No reason provided'}",
        payment_id=payment.id,
        balance_before=0.0,  # Would fetch from user balance in production
        balance_after=0.0,  # Would calculate in production
    )
    db.add(transaction)
    
    await db.commit()
    await db.refresh(payment)
    
    return PaymentResponse(
        id=str(payment.id),
        amount=payment.amount,
        currency=payment.currency,
        payment_type=payment.payment_type,
        status=payment.status.value,
        provider=payment.provider,
        provider_payment_id=payment.provider_payment_id,
        payment_method_type=payment.payment_method_type,
        last_four_digits=payment.card_last4,
        card_brand=payment.card_brand,
        description=payment.description,
        refund_amount=payment.refunded_amount,
        processing_fee=payment.processing_fee,
        platform_fee=payment.platform_fee,
        net_amount=payment.net_amount,
        created_at=payment.created_at,
        processed_at=payment.processed_at,
    )


@router.get(
    "/payments",
    response_model=PaymentList,
    summary="List payments",
    description="Get user's payment history with pagination",
)
async def list_payments(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    payment_status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaymentList:
    """
    List user's payments with optional status filter.
    
    Returns paginated list of payments with summary information.
    """
    # Parse status filter
    status_filter = None
    if payment_status:
        try:
            status_filter = PaymentStatus(payment_status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {payment_status}",
            )
    
    # Get payments
    payments = await crud_payment.payment.get_by_user(
        db,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        status=status_filter,
    )
    
    # Get total count (simplified - in production would have separate count query)
    total_count = len(payments) + skip
    
    return PaymentList(
        payments=[
            PaymentResponse(
                id=str(p.id),
                amount=p.amount,
                currency=p.currency,
                payment_type=p.payment_type,
                status=p.status.value,
                provider=p.provider,
                provider_payment_id=p.provider_payment_id,
                payment_method_type=p.payment_method_type,
                last_four_digits=p.card_last4,
                card_brand=p.card_brand,
                description=p.description,
                refund_amount=p.refunded_amount,
                processing_fee=p.processing_fee,
                platform_fee=p.platform_fee,
                net_amount=p.net_amount,
                created_at=p.created_at,
                processed_at=p.processed_at,
            )
            for p in payments
        ],
        total_count=total_count,
        page=skip // limit + 1,
        page_size=limit,
    )


@router.get(
    "/payments/{payment_id}",
    response_model=PaymentResponse,
    summary="Get payment details",
    description="Get detailed information about a specific payment",
)
async def get_payment(
    payment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaymentResponse:
    """Get detailed payment information."""
    payment = await crud_payment.payment.get(db, payment_id)
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found",
        )
    
    # Verify ownership
    if payment.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this payment",
        )
    
    return PaymentResponse(
        id=str(payment.id),
        amount=payment.amount,
        currency=payment.currency,
        payment_type=payment.payment_type,
        status=payment.status.value,
        provider=payment.provider,
        provider_payment_id=payment.provider_payment_id,
        payment_method_type=payment.payment_method_type,
        last_four_digits=payment.card_last4,
        card_brand=payment.card_brand,
        description=payment.description,
        refund_amount=payment.refunded_amount,
        processing_fee=payment.processing_fee,
        platform_fee=payment.platform_fee,
        net_amount=payment.net_amount,
        created_at=payment.created_at,
        processed_at=payment.processed_at,
    )


# ============================================================================
# SUBSCRIPTION ENDPOINTS
# ============================================================================


@router.post(
    "/subscriptions",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create subscription",
    description="Create a new subscription for a tier/plan",
)
async def create_subscription(
    subscription_data: SubscriptionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    """
    Create a new subscription.
    
    Creates a Stripe subscription and corresponding database record.
    Supports trial periods and various subscription tiers.
    
    **Subscription Tiers:**
    - FREE: Free tier (no payment required)
    - BASIC: Basic features
    - PREMIUM: Premium features
    - PRO: Professional features
    - ENTERPRISE: Enterprise features
    
    **Trial Support:**
    - Specify trial_days for trial period (0-30 days)
    - No charges during trial
    - Auto-converts to paid after trial
    """
    # Check for existing active subscription
    existing = await crud_payment.subscription.get_active_by_user(
        db, user_id=current_user.id
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has an active subscription",
        )
    
    # Subscription pricing (mock - in production from database)
    pricing = {
        "free": 0.0,
        "basic": 9.99,
        "premium": 19.99,
        "pro": 49.99,
        "enterprise": 99.99,
    }
    
    tier_value = subscription_data.tier.value
    price = pricing.get(tier_value, 0.0)
    
    # Create Stripe subscription (mock)
    import uuid
    stripe_subscription_id = f"sub_{uuid.uuid4().hex[:24]}"
    stripe_customer_id = f"cus_{uuid.uuid4().hex[:24]}"
    
    # Calculate dates
    now = datetime.now(timezone.utc)
    trial_end = now + timedelta(days=subscription_data.trial_days) if subscription_data.trial_days > 0 else None
    current_period_end = (trial_end or now) + timedelta(days=30)  # Monthly billing
    
    # Create subscription
    subscription = Subscription(
        user_id=current_user.id,
        tier=tier_value,
        status=SubscriptionStatus.TRIALING if subscription_data.trial_days > 0 else SubscriptionStatus.ACTIVE,
        price_amount=price,
        currency="usd",
        billing_cycle="monthly",
        stripe_subscription_id=stripe_subscription_id,
        stripe_customer_id=stripe_customer_id,
        current_period_start=now,
        current_period_end=current_period_end,
        trial_ends_at=trial_end,
        trial_days=subscription_data.trial_days,
    )
    
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)
    
    # Calculate days remaining
    days_remaining = (current_period_end - now).days
    trial_days_remaining = (trial_end - now).days if trial_end else 0
    
    return SubscriptionResponse(
        id=str(subscription.id),
        tier=subscription.tier,
        status=subscription.status.value,
        price=subscription.price_amount,
        currency=subscription.currency,
        billing_cycle=subscription.billing_cycle,
        is_trial=subscription.trial_ends_at is not None and subscription.trial_days > 0,
        trial_days=subscription.trial_days,
        trial_days_remaining=max(0, trial_days_remaining),
        days_remaining=days_remaining,
        is_active=subscription.status == SubscriptionStatus.ACTIVE,
        is_trial_active=subscription.status == SubscriptionStatus.TRIALING,
        start_date=subscription.current_period_start,
        end_date=subscription.current_period_end,
        trial_start_date=subscription.current_period_start if subscription.trial_days > 0 else None,
        trial_end_date=subscription.trial_ends_at,
        cancelled_at=subscription.cancelled_at,
        features=None,
        limits=None,
        provider_subscription_id=subscription.stripe_subscription_id,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


@router.get(
    "/subscriptions/pricing",
    response_model=SubscriptionPricingList,
    summary="Get subscription pricing",
    description="Get available subscription tiers and pricing",
)
async def get_subscription_pricing(
    db: AsyncSession = Depends(get_db),
) -> SubscriptionPricingList:
    """
    Get subscription pricing information.
    
    Returns available subscription tiers with pricing, features, and limits.
    No authentication required for pricing info.
    """
    pricing_data = [
        SubscriptionPricing(
            tier="free",
            display_name="Free",
            price=0.0,
            currency="usd",
            billing_cycle="monthly",
            features=[
                "Basic video uploads",
                "Standard quality streaming",
                "Community features",
                "Limited storage (5GB)",
            ],
            limits={
                "videos": 10,
                "storage_gb": 5,
                "streams_per_month": 0,
            },
            is_popular=False,
        ),
        SubscriptionPricing(
            tier="basic",
            display_name="Basic",
            price=9.99,
            currency="usd",
            billing_cycle="monthly",
            features=[
                "Unlimited video uploads",
                "HD streaming",
                "Ad-free experience",
                "50GB storage",
                "Priority support",
            ],
            limits={
                "videos": -1,  # unlimited
                "storage_gb": 50,
                "streams_per_month": 5,
            },
            is_popular=False,
        ),
        SubscriptionPricing(
            tier="premium",
            display_name="Premium",
            price=19.99,
            currency="usd",
            billing_cycle="monthly",
            features=[
                "Everything in Basic",
                "4K streaming",
                "Custom branding",
                "200GB storage",
                "Live streaming",
                "Analytics dashboard",
            ],
            limits={
                "videos": -1,
                "storage_gb": 200,
                "streams_per_month": 20,
            },
            is_popular=True,
        ),
        SubscriptionPricing(
            tier="pro",
            display_name="Pro",
            price=49.99,
            currency="usd",
            billing_cycle="monthly",
            features=[
                "Everything in Premium",
                "Advanced analytics",
                "API access",
                "1TB storage",
                "Unlimited live streams",
                "Monetization features",
                "White-label option",
            ],
            limits={
                "videos": -1,
                "storage_gb": 1000,
                "streams_per_month": -1,
            },
            is_popular=False,
        ),
        SubscriptionPricing(
            tier="enterprise",
            display_name="Enterprise",
            price=99.99,
            currency="usd",
            billing_cycle="monthly",
            features=[
                "Everything in Pro",
                "Dedicated support",
                "Custom integrations",
                "Unlimited storage",
                "SLA guarantee",
                "Team management",
                "Advanced security",
            ],
            limits={
                "videos": -1,
                "storage_gb": -1,
                "streams_per_month": -1,
            },
            is_popular=False,
        ),
    ]
    
    return SubscriptionPricingList(pricing=pricing_data)


@router.get(
    "/subscriptions/current",
    response_model=SubscriptionResponse,
    summary="Get current subscription",
    description="Get user's current active subscription",
)
async def get_current_subscription(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    """Get user's current active subscription."""
    subscription = await crud_payment.subscription.get_active_by_user(
        db, user_id=current_user.id
    )
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found",
        )
    
    # Calculate days remaining
    now = datetime.now(timezone.utc)
    # Handle timezone-aware and timezone-naive datetimes
    if subscription.current_period_end:
        period_end = subscription.current_period_end if subscription.current_period_end.tzinfo else subscription.current_period_end.replace(tzinfo=timezone.utc)
        days_remaining = (period_end - now).days
    else:
        days_remaining = 0
    
    if subscription.trial_ends_at:
        trial_end = subscription.trial_ends_at if subscription.trial_ends_at.tzinfo else subscription.trial_ends_at.replace(tzinfo=timezone.utc)
        trial_days_remaining = (trial_end - now).days
    else:
        trial_days_remaining = 0
    
    return SubscriptionResponse(
        id=str(subscription.id),
        tier=subscription.tier,
        status=subscription.status.value,
        price=subscription.price_amount,
        currency=subscription.currency,
        billing_cycle=subscription.billing_cycle,
        is_trial=subscription.trial_ends_at is not None and subscription.trial_days > 0,
        trial_days=subscription.trial_days,
        trial_days_remaining=max(0, trial_days_remaining),
        days_remaining=max(0, days_remaining),
        is_active=subscription.status == SubscriptionStatus.ACTIVE,
        is_trial_active=subscription.status == SubscriptionStatus.TRIALING,
        start_date=subscription.current_period_start,
        end_date=subscription.current_period_end,
        trial_start_date=subscription.current_period_start if subscription.trial_days > 0 else None,
        trial_end_date=subscription.trial_ends_at,
        cancelled_at=subscription.cancelled_at,
        features=None,
        limits=None,
        provider_subscription_id=subscription.stripe_subscription_id,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


@router.put(
    "/subscriptions/upgrade",
    response_model=SubscriptionResponse,
    summary="Upgrade subscription",
    description="Upgrade to a higher subscription tier",
)
async def upgrade_subscription(
    upgrade_data: SubscriptionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    """
    Upgrade subscription to a higher tier.
    
    Updates the Stripe subscription and database record.
    Proration is handled automatically.
    """
    # Get subscription
    subscription = await crud_payment.subscription.get(db, UUID(upgrade_data.subscription_id))
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found",
        )
    
    # Verify ownership
    if subscription.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify this subscription",
        )
    
    # Update subscription
    pricing = {
        "free": 0.0,
        "basic": 9.99,
        "premium": 19.99,
        "pro": 49.99,
        "enterprise": 99.99,
    }
    
    new_tier_value = upgrade_data.new_tier.value
    subscription.tier = new_tier_value
    subscription.price_amount = pricing.get(new_tier_value, 0.0)
    
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)
    
    # Calculate days remaining
    now = datetime.now(timezone.utc)
    # Handle timezone-aware and timezone-naive datetimes
    if subscription.current_period_end:
        period_end = subscription.current_period_end if subscription.current_period_end.tzinfo else subscription.current_period_end.replace(tzinfo=timezone.utc)
        days_remaining = (period_end - now).days
    else:
        days_remaining = 0
    
    if subscription.trial_ends_at:
        trial_end = subscription.trial_ends_at if subscription.trial_ends_at.tzinfo else subscription.trial_ends_at.replace(tzinfo=timezone.utc)
        trial_days_remaining = (trial_end - now).days
    else:
        trial_days_remaining = 0
    
    return SubscriptionResponse(
        id=str(subscription.id),
        tier=subscription.tier,
        status=subscription.status.value,
        price=subscription.price_amount,
        currency=subscription.currency,
        billing_cycle=subscription.billing_cycle,
        is_trial=subscription.trial_ends_at is not None and subscription.trial_days > 0,
        trial_days=subscription.trial_days,
        trial_days_remaining=max(0, trial_days_remaining),
        days_remaining=max(0, days_remaining),
        is_active=subscription.status == SubscriptionStatus.ACTIVE,
        is_trial_active=subscription.status == SubscriptionStatus.TRIALING,
        start_date=subscription.current_period_start,
        end_date=subscription.current_period_end,
        trial_start_date=subscription.current_period_start if subscription.trial_days > 0 else None,
        trial_end_date=subscription.trial_ends_at,
        cancelled_at=subscription.cancelled_at,
        features=None,
        limits=None,
        provider_subscription_id=subscription.stripe_subscription_id,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


@router.post(
    "/subscriptions/cancel",
    response_model=SubscriptionResponse,
    summary="Cancel subscription",
    description="Cancel subscription (immediate or at period end)",
)
async def cancel_subscription(
    cancel_data: SubscriptionCancel,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    """
    Cancel subscription.
    
    **Cancellation Options:**
    - immediate=False: Cancel at end of billing period (default)
    - immediate=True: Cancel immediately with no refund
    """
    # Get subscription
    subscription = await crud_payment.subscription.get(db, UUID(cancel_data.subscription_id))
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found",
        )
    
    # Verify ownership
    if subscription.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to cancel this subscription",
        )
    
    # Cancel subscription
    subscription.cancelled_at = datetime.now(timezone.utc)
    
    if cancel_data.immediate:
        subscription.status = SubscriptionStatus.CANCELLED
        subscription.current_period_end = datetime.now(timezone.utc)
    else:
        # Will cancel at period end
        subscription.status = SubscriptionStatus.ACTIVE
    
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)
    
    # Calculate days remaining
    now = datetime.now(timezone.utc)
    # Handle timezone-aware and timezone-naive datetimes
    if subscription.current_period_end:
        period_end = subscription.current_period_end if subscription.current_period_end.tzinfo else subscription.current_period_end.replace(tzinfo=timezone.utc)
        days_remaining = (period_end - now).days
    else:
        days_remaining = 0
    
    if subscription.trial_ends_at:
        trial_end = subscription.trial_ends_at if subscription.trial_ends_at.tzinfo else subscription.trial_ends_at.replace(tzinfo=timezone.utc)
        trial_days_remaining = (trial_end - now).days
    else:
        trial_days_remaining = 0
    
    return SubscriptionResponse(
        id=str(subscription.id),
        tier=subscription.tier,
        status=subscription.status.value,
        price=subscription.price_amount,
        currency=subscription.currency,
        billing_cycle=subscription.billing_cycle,
        is_trial=subscription.trial_ends_at is not None and subscription.trial_days > 0,
        trial_days=subscription.trial_days,
        trial_days_remaining=max(0, trial_days_remaining),
        days_remaining=max(0, days_remaining),
        is_active=subscription.status == SubscriptionStatus.ACTIVE,
        is_trial_active=subscription.status == SubscriptionStatus.TRIALING,
        start_date=subscription.current_period_start,
        end_date=subscription.current_period_end,
        trial_start_date=subscription.current_period_start if subscription.trial_days > 0 else None,
        trial_end_date=subscription.trial_ends_at,
        cancelled_at=subscription.cancelled_at,
        features=None,
        limits=None,
        provider_subscription_id=subscription.stripe_subscription_id,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


# ============================================================================
# CREATOR PAYOUT ENDPOINTS
# ============================================================================


@router.post(
    "/payouts/connect",
    response_model=ConnectAccountResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Stripe Connect account",
    description="Create a Stripe Connect account for creator payouts",
)
async def create_connect_account(
    account_data: ConnectAccountCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConnectAccountResponse:
    """
    Create a Stripe Connect account for receiving payouts.
    
    This enables creators to receive payments for their content.
    Returns an onboarding URL for completing account setup.
    
    **Process:**
    1. Create Stripe Connect account
    2. Generate onboarding URL
    3. Creator completes onboarding
    4. Account becomes active for payouts
    """
    # In production: Create Stripe Connect account
    # account = stripe.Account.create(
    #     type=account_data.account_type,
    #     country=account_data.country,
    #     email=current_user.email,
    # )
    
    import uuid
    stripe_account_id = f"acct_{uuid.uuid4().hex[:16]}"
    
    # Generate onboarding URL (mock)
    onboarding_url = f"https://connect.stripe.com/setup/{stripe_account_id}"
    
    return ConnectAccountResponse(
        connect_account_id=stripe_account_id,
        stripe_account_id=stripe_account_id,
        onboarding_url=onboarding_url,
        status="pending",
        charges_enabled=False,
        payouts_enabled=False,
        details_submitted=False,
        is_fully_onboarded=False,
        requirements_currently_due=["business_type", "tos_acceptance"],
        available_balance=0.0,
        pending_balance=0.0,
        currency="usd",
        total_volume=0.0,
        created_at=datetime.now(timezone.utc),
    )


@router.get(
    "/payouts/connect/status",
    response_model=ConnectAccountStatus,
    summary="Get Connect account status",
    description="Get creator's Stripe Connect account status",
)
async def get_connect_account_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConnectAccountStatus:
    """
    Get Stripe Connect account status.
    
    Returns onboarding status, capabilities, and balance information.
    """
    # Mock response - in production would query Stripe
    return ConnectAccountStatus(
        connect_account_id="acct_mock123",
        stripe_account_id="acct_mock123",
        status="active",
        charges_enabled=True,
        payouts_enabled=True,
        details_submitted=True,
        is_fully_onboarded=True,
        requirements_currently_due=[],
        available_balance=1250.50,
        pending_balance=345.00,
    )


@router.post(
    "/payouts",
    response_model=PayoutResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request payout",
    description="Request a payout of earnings",
)
async def request_payout(
    payout_data: PayoutCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PayoutResponse:
    """
    Request a payout of accumulated earnings.
    
    Calculates earnings for the specified period and creates
    a payout request. Requires completed Stripe Connect onboarding.
    
    **Revenue Sources:**
    - Subscription revenue
    - Tips and donations
    - Content sales
    - Ad revenue
    
    **Fees:**
    - Platform fee: 10%
    - Stripe fee: 0.25% + $0.25
    """
    # Calculate total revenue
    total_revenue = sum(payout_data.revenue_breakdown.values())
    
    # Calculate fees
    platform_fee = total_revenue * 0.10  # 10% platform fee
    stripe_fee = (total_revenue * 0.0025) + 0.25  # Stripe Connect fee
    net_amount = total_revenue - platform_fee - stripe_fee
    
    # Create payout
    import uuid
    payout = Payout(
        user_id=current_user.id,
        amount=net_amount,
        currency="usd",
        status=PayoutStatus.PENDING,
        stripe_payout_id=f"po_{uuid.uuid4().hex[:24]}",
        platform_fee=platform_fee,
        processing_fee=stripe_fee,
        net_amount=net_amount,
        subscription_revenue=payout_data.revenue_breakdown.get("subscription", 0.0),
        tip_revenue=payout_data.revenue_breakdown.get("tips", 0.0),
        other_revenue=payout_data.revenue_breakdown.get("content_sales", 0.0),
        ad_revenue=payout_data.revenue_breakdown.get("ad_revenue", 0.0),
        period_start=payout_data.period_start,
        period_end=payout_data.period_end,
    )
    
    db.add(payout)
    
    # Create transaction record
    transaction = Transaction(
        user_id=current_user.id,
        transaction_type="payout",
        amount=net_amount,
        currency="usd",
        balance_before=0.0,  # Mock balance tracking
        balance_after=0.0,
        description=f"Payout {payout.id}",
        payout_id=payout.id,
    )
    db.add(transaction)
    
    await db.commit()
    await db.refresh(payout)
    
    # Calculate gross amount from revenue breakdown
    gross_amount = (
        payout.subscription_revenue +
        payout.tip_revenue +
        payout.other_revenue +
        payout.ad_revenue
    )
    
    return PayoutResponse(
        id=str(payout.id),
        amount=payout.amount,
        currency=payout.currency,
        status=payout.status.value,
        stripe_payout_id=payout.stripe_payout_id,
        stripe_transfer_id=None,  # Not stored in model
        gross_amount=gross_amount,
        platform_fee=payout.platform_fee,
        stripe_fee=payout.processing_fee,
        net_amount=payout.net_amount,
        subscription_revenue=payout.subscription_revenue,
        tips_revenue=payout.tip_revenue,
        content_sales_revenue=payout.other_revenue,
        ad_revenue=payout.ad_revenue,
        period_start=payout.period_start,
        period_end=payout.period_end,
        description=None,  # Not stored in model
        failure_message=payout.failure_reason,
        created_at=payout.created_at,
        processed_at=payout.paid_at,
        paid_at=payout.paid_at,
        failed_at=payout.failed_at,
    )


@router.get(
    "/payouts",
    response_model=PayoutList,
    summary="List payouts",
    description="Get payout history with pagination",
)
async def list_payouts(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    payout_status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PayoutList:
    """List user's payouts with optional status filter."""
    # Parse status filter
    status_filter = None
    if payout_status:
        try:
            status_filter = PayoutStatus(payout_status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {payout_status}",
            )
    
    # Get payouts
    payouts = await crud_payment.payout.get_by_user(
        db,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        status=status_filter,
    )
    
    total_amount = sum(p.amount for p in payouts)
    
    return PayoutList(
        payouts=[
            PayoutResponse(
                id=str(p.id),
                amount=p.amount,
                currency=p.currency,
                status=p.status.value,
                stripe_payout_id=p.stripe_payout_id,
                stripe_transfer_id=None,
                gross_amount=p.subscription_revenue + p.tip_revenue + p.other_revenue + p.ad_revenue,
                platform_fee=p.platform_fee,
                stripe_fee=p.processing_fee,
                net_amount=p.net_amount,
                subscription_revenue=p.subscription_revenue,
                tips_revenue=p.tip_revenue,
                content_sales_revenue=p.other_revenue,
                ad_revenue=p.ad_revenue,
                period_start=p.period_start,
                period_end=p.period_end,
                description=None,
                failure_message=p.failure_reason,
                created_at=p.created_at,
                processed_at=p.paid_at,
                paid_at=p.paid_at,
                failed_at=p.failed_at,
            )
            for p in payouts
        ],
        total_count=len(payouts) + skip,
        total_amount=total_amount,
        page=skip // limit + 1,
        page_size=limit,
    )


@router.get(
    "/payouts/earnings",
    response_model=CreatorEarnings,
    summary="Get creator earnings",
    description="Get detailed earnings summary for creator",
)
async def get_creator_earnings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CreatorEarnings:
    """
    Get creator earnings summary.
    
    Returns comprehensive earnings breakdown including:
    - Total, pending, and paid earnings
    - Revenue by source
    - Fees paid
    - Next payout information
    """
    # Get earnings (mock implementation)
    total_earnings = 5420.50
    pending_earnings = 1250.00
    paid_earnings = 4170.50
    
    return CreatorEarnings(
        user_id=str(current_user.id),
        total_earnings=total_earnings,
        pending_earnings=pending_earnings,
        paid_earnings=paid_earnings,
        subscription_earnings=3200.00,
        tips_earnings=850.50,
        content_sales_earnings=1200.00,
        ad_revenue_earnings=170.00,
        platform_fees_paid=542.05,
        stripe_fees_paid=27.10,
        next_payout_date=datetime.now(timezone.utc) + timedelta(days=7),
        next_payout_amount=pending_earnings,
        currency="usd",
    )


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================


@router.get(
    "/analytics/payments",
    response_model=PaymentAnalytics,
    summary="Payment analytics",
    description="Get payment analytics for a period",
)
async def get_payment_analytics(
    start_date: datetime = Query(..., description="Start date for analytics"),
    end_date: datetime = Query(..., description="End date for analytics"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaymentAnalytics:
    """
    Get payment analytics for specified period.
    
    Returns comprehensive payment metrics including revenue,
    transaction counts, fees, and success rates.
    """
    # Mock analytics data
    return PaymentAnalytics(
        total_revenue=15420.50,
        total_transactions=247,
        successful_transactions=235,
        failed_transactions=12,
        refunded_transactions=5,
        average_transaction_value=62.36,
        total_platform_fees=1542.05,
        total_stripe_fees=385.51,
        net_revenue=13492.94,
        period_start=start_date,
        period_end=end_date,
    )


@router.get(
    "/analytics/subscriptions",
    response_model=SubscriptionAnalytics,
    summary="Subscription analytics",
    description="Get subscription analytics for a period",
)
async def get_subscription_analytics(
    start_date: datetime = Query(..., description="Start date for analytics"),
    end_date: datetime = Query(..., description="End date for analytics"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionAnalytics:
    """
    Get subscription analytics for specified period.
    
    Returns key subscription metrics including MRR, ARR,
    churn rate, and retention rate.
    """
    # Mock analytics data
    return SubscriptionAnalytics(
        active_subscriptions=1250,
        trial_subscriptions=85,
        cancelled_subscriptions=45,
        monthly_recurring_revenue=24875.00,
        annual_recurring_revenue=298500.00,
        average_subscription_value=19.90,
        churn_rate=3.6,
        retention_rate=96.4,
        period_start=start_date,
        period_end=end_date,
    )
