"""
Stripe Subscription API Routes.

Subscription management: create, update, cancel subscriptions.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.auth.api.auth import get_current_active_user
from app.auth.models.user import User
from app.payments.schemas.payment import (
    SubscriptionCreate,
    SubscriptionUpdate,
    SubscriptionCancel,
    SubscriptionResponse,
    SubscriptionPricingList,
    SubscriptionPricing,
    SubscriptionAnalytics,
)
from app.payments.services.stripe_payment_service import StripePaymentService


router = APIRouter(prefix="/subscriptions", tags=["Subscriptions"])


@router.post(
    "/",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Subscription",
    description="Create a new subscription for the authenticated user."
)
async def create_subscription(
    request: SubscriptionCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> SubscriptionResponse:
    """
    Create subscription.
    
    Args:
        request: Subscription creation request
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Created subscription details
    """
    service = StripePaymentService(db)
    
    try:
        subscription = await service.create_subscription(
            user_id=current_user.id,
            tier=request.tier.value,
            payment_method_id=request.payment_method_id,
            trial_days=request.trial_days
        )
        
        return SubscriptionResponse.from_orm(subscription)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create subscription: {str(e)}"
        )


@router.get(
    "/{subscription_id}",
    response_model=SubscriptionResponse,
    summary="Get Subscription",
    description="Retrieve details for a specific subscription."
)
async def get_subscription(
    subscription_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> SubscriptionResponse:
    """
    Get subscription details.
    
    Args:
        subscription_id: Subscription ID
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Subscription details
    """
    from sqlalchemy import select
    from app.auth.models.subscription import Subscription
    
    result = await db.execute(
        select(Subscription).where(
            Subscription.id == subscription_id,
            Subscription.user_id == current_user.id
        )
    )
    subscription = result.scalar_one_or_none()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found"
        )
    
    return SubscriptionResponse.from_orm(subscription)


@router.get(
    "/me/current",
    response_model=SubscriptionResponse,
    summary="Get Current Subscription",
    description="Get the authenticated user's current subscription."
)
async def get_current_subscription(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> SubscriptionResponse:
    """
    Get current user subscription.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Current subscription details
    """
    from sqlalchemy import select
    from app.auth.models.subscription import Subscription, SubscriptionStatus
    
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.status.in_([
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.TRIAL,
                SubscriptionStatus.PAST_DUE
            ])
        ).order_by(Subscription.created_at.desc())
    )
    subscription = result.scalar_one_or_none()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
    
    return SubscriptionResponse.from_orm(subscription)


@router.put(
    "/{subscription_id}",
    response_model=SubscriptionResponse,
    summary="Update Subscription",
    description="Update subscription tier (upgrade/downgrade)."
)
async def update_subscription(
    subscription_id: UUID,
    request: SubscriptionUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> SubscriptionResponse:
    """
    Update subscription tier.
    
    Args:
        subscription_id: Subscription ID
        request: Subscription update request
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Updated subscription details
    """
    service = StripePaymentService(db)
    
    try:
        subscription = await service.update_subscription(
            subscription_id=subscription_id,
            user_id=current_user.id,
            new_tier=request.new_tier.value
        )
        
        return SubscriptionResponse.from_orm(subscription)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update subscription: {str(e)}"
        )


@router.delete(
    "/{subscription_id}",
    response_model=SubscriptionResponse,
    summary="Cancel Subscription",
    description="Cancel a subscription (immediate or at period end)."
)
async def cancel_subscription(
    subscription_id: UUID,
    request: Optional[SubscriptionCancel] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> SubscriptionResponse:
    """
    Cancel subscription.
    
    Args:
        subscription_id: Subscription ID
        request: Cancellation options (immediate vs at period end)
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Cancelled subscription details
    """
    service = StripePaymentService(db)
    
    immediate = request.immediate if request else False
    
    try:
        subscription = await service.cancel_subscription(
            subscription_id=subscription_id,
            user_id=current_user.id,
            immediate=immediate
        )
        
        return SubscriptionResponse.from_orm(subscription)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to cancel subscription: {str(e)}"
        )


@router.get(
    "/pricing/list",
    response_model=SubscriptionPricingList,
    summary="Get Subscription Pricing",
    description="Get all available subscription tiers with pricing and features."
)
async def get_subscription_pricing(
    db: AsyncSession = Depends(get_db)
) -> SubscriptionPricingList:
    """
    Get subscription pricing for all tiers.
    
    Args:
        db: Database session
        
    Returns:
        List of subscription pricing options
    """
    service = StripePaymentService(db)
    
    # Get pricing for all tiers
    tiers = ["free", "basic", "premium", "pro"]
    pricing_list = []
    
    for tier in tiers:
        pricing_info = await service._get_subscription_pricing(tier)
        
        pricing_list.append(SubscriptionPricing(
            tier=tier,
            display_name=pricing_info["display_name"],
            price=pricing_info["price"],
            currency=pricing_info["currency"],
            billing_cycle=pricing_info["billing_cycle"],
            features=pricing_info["features"],
            limits=pricing_info["limits"],
            is_popular=(tier == "premium")  # Mark premium as popular
        ))
    
    return SubscriptionPricingList(pricing=pricing_list)


@router.get(
    "/analytics/summary",
    response_model=SubscriptionAnalytics,
    summary="Get Subscription Analytics",
    description="Get subscription analytics for admin users."
)
async def get_subscription_analytics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> SubscriptionAnalytics:
    """
    Get subscription analytics.
    
    Note: This endpoint should be restricted to admin users in production.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Subscription analytics data
    """
    from sqlalchemy import select, func
    from app.auth.models.subscription import Subscription, SubscriptionStatus
    from datetime import datetime, timedelta
    
    # Get data for last 30 days
    period_start = datetime.utcnow() - timedelta(days=30)
    period_end = datetime.utcnow()
    
    # Active subscriptions
    active_result = await db.execute(
        select(func.count(Subscription.id))
        .where(Subscription.status == SubscriptionStatus.ACTIVE)
    )
    active_count = active_result.scalar()
    
    # Trial subscriptions
    trial_result = await db.execute(
        select(func.count(Subscription.id))
        .where(Subscription.status == SubscriptionStatus.TRIAL)
    )
    trial_count = trial_result.scalar()
    
    # Cancelled in period
    cancelled_result = await db.execute(
        select(func.count(Subscription.id))
        .where(
            Subscription.status == SubscriptionStatus.CANCELLED,
            Subscription.cancelled_at >= period_start
        )
    )
    cancelled_count = cancelled_result.scalar()
    
    # Monthly recurring revenue (MRR)
    mrr_result = await db.execute(
        select(func.sum(Subscription.price))
        .where(
            Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.PAST_DUE]),
            Subscription.billing_cycle == "monthly"
        )
    )
    mrr = float(mrr_result.scalar() or 0)
    
    # Calculate churn and retention
    total_subs = active_count + trial_count
    churn_rate = (cancelled_count / total_subs * 100) if total_subs > 0 else 0
    retention_rate = 100 - churn_rate
    
    return SubscriptionAnalytics(
        active_subscriptions=active_count,
        trial_subscriptions=trial_count,
        cancelled_subscriptions=cancelled_count,
        monthly_recurring_revenue=mrr,
        annual_recurring_revenue=mrr * 12,
        average_subscription_value=mrr / total_subs if total_subs > 0 else 0,
        churn_rate=churn_rate,
        retention_rate=retention_rate,
        period_start=period_start,
        period_end=period_end
    )
