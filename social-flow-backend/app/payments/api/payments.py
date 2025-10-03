"""
Payments endpoints.

This module contains all payment and monetization-related API endpoints.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from typing import List, Optional
from uuid import UUID

from app.models.user import User
from app.auth.api.auth import get_current_active_user
from app.payments.services.payments_service import payments_service

router = APIRouter()
@router.post("/create-intent")
async def create_payment_intent(
    payload: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a Stripe payment intent. Minimal endpoint for integration tests.

    Tests patch stripe.PaymentIntent.create; this endpoint simply forwards to the service.
    """
    amount = payload.get("amount")
    currency = payload.get("currency", "USD")
    result = await payments_service.create_payment_intent(str(current_user.id), amount, currency)
    return result



@router.post("/process")
async def process_payment(
    payment_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Process a payment."""
    try:
        result = await payments_service.process_payment(payment_data)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process payment")


@router.get("/history")
async def get_payment_history(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get payment history."""
    try:
        result = await payments_service.get_payment_history(
            user_id=str(current_user.id),
            limit=limit
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get payment history")


@router.get("/{payment_id}")
async def get_payment_status(
    payment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get payment status."""
    try:
        result = await payments_service.get_payment_status(payment_id)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get payment status")


# Enhanced monetization endpoints from Kotlin service

@router.post("/subscriptions")
async def process_subscription(
    subscription_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Process subscription payment."""
    try:
        result = await payments_service.process_subscription(
            user_id=str(current_user.id),
            subscription_data=subscription_data
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process subscription")


@router.get("/subscriptions/plans")
async def get_subscription_plans(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get available subscription plans."""
    try:
        result = await payments_service.get_subscription_plans()
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get subscription plans")


@router.put("/subscriptions/{subscription_id}")
async def update_subscription(
    subscription_id: str,
    new_plan: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update user subscription plan."""
    try:
        result = await payments_service.update_subscription(subscription_id, new_plan)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update subscription")


@router.delete("/subscriptions/{subscription_id}")
async def cancel_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Cancel user subscription."""
    try:
        result = await payments_service.cancel_subscription(subscription_id)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")


@router.post("/donations")
async def process_donation(
    donation_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Process donation payment."""
    try:
        result = await payments_service.process_donation(
            user_id=str(current_user.id),
            donation_data=donation_data
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process donation")


@router.get("/creators/{creator_id}/earnings")
async def get_creator_earnings(
    creator_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get creator earnings summary."""
    try:
        result = await payments_service.get_creator_earnings(
            creator_id=creator_id,
            time_range=time_range
        )
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get creator earnings")


@router.post("/creators/{creator_id}/payouts")
async def schedule_creator_payout(
    creator_id: str,
    amount: float,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Schedule creator payout."""
    try:
        result = await payments_service.schedule_creator_payout(creator_id, amount)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to schedule creator payout")


@router.get("/creators/{creator_id}/tax-reports")
async def generate_tax_report(
    creator_id: str,
    period: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Generate tax report for creator."""
    try:
        result = await payments_service.generate_tax_report(creator_id, period)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate tax report")


@router.get("/analytics/revenue")
async def get_revenue_analytics(
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get revenue analytics for the platform."""
    try:
        result = await payments_service.get_revenue_analytics(time_range=time_range)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get revenue analytics")


@router.post("/refunds")
async def process_refund(
    payment_id: str,
    reason: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Process payment refund."""
    try:
        result = await payments_service.process_refund(payment_id, reason)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process refund")
