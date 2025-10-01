"""
Stripe Payment API Routes.

One-time payments: tips, content purchases, donations.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.auth.models.user import User
from app.payments.schemas.payment import (
    PaymentIntentCreate,
    PaymentIntentResponse,
    PaymentRefund,
    PaymentResponse,
    PaymentList,
    PaymentAnalytics,
)
from app.payments.services.stripe_payment_service import StripePaymentService


router = APIRouter(prefix="/payments", tags=["Payments"])


@router.post(
    "/intent",
    response_model=PaymentIntentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Payment Intent",
    description="Create a Stripe payment intent for one-time payments (tips, purchases)."
)
async def create_payment_intent(
    request: PaymentIntentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentIntentResponse:
    """
    Create payment intent for one-time payment.
    
    Args:
        request: Payment intent creation request
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Payment intent with client_secret for frontend
    """
    service = StripePaymentService(db)
    
    try:
        payment, client_secret = await service.create_payment_intent(
            user_id=current_user.id,
            amount=request.amount,
            currency=request.currency,
            payment_type=request.payment_type,
            description=request.description,
            metadata=request.metadata
        )
        
        return PaymentIntentResponse(
            payment_id=str(payment.id),
            client_secret=client_secret,
            stripe_payment_intent_id=payment.provider_payment_id,
            amount=float(payment.amount),
            currency=payment.currency,
            status=payment.status.value
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create payment intent: {str(e)}"
        )


@router.post(
    "/{payment_id}/confirm",
    response_model=PaymentResponse,
    summary="Confirm Payment",
    description="Confirm payment after successful charge by Stripe."
)
async def confirm_payment(
    payment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentResponse:
    """
    Confirm payment after successful charge.
    
    Args:
        payment_id: Payment ID to confirm
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Updated payment details
    """
    service = StripePaymentService(db)
    
    try:
        payment = await service.confirm_payment(
            payment_id=payment_id,
            user_id=current_user.id
        )
        
        return PaymentResponse.from_orm(payment)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to confirm payment: {str(e)}"
        )


@router.post(
    "/{payment_id}/refund",
    response_model=PaymentResponse,
    summary="Refund Payment",
    description="Refund a payment (full or partial)."
)
async def refund_payment(
    payment_id: UUID,
    request: Optional[PaymentRefund] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentResponse:
    """
    Refund payment (full or partial).
    
    Args:
        payment_id: Payment ID to refund
        request: Refund request with optional amount and reason
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Updated payment details with refund information
    """
    service = StripePaymentService(db)
    
    try:
        payment = await service.refund_payment(
            payment_id=payment_id,
            user_id=current_user.id,
            amount=request.amount if request else None,
            reason=request.reason if request else None
        )
        
        return PaymentResponse.from_orm(payment)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to refund payment: {str(e)}"
        )


@router.get(
    "/{payment_id}",
    response_model=PaymentResponse,
    summary="Get Payment Details",
    description="Retrieve details for a specific payment."
)
async def get_payment(
    payment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentResponse:
    """
    Get payment details.
    
    Args:
        payment_id: Payment ID
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Payment details
    """
    from sqlalchemy import select
    from app.payments.models.payment import Payment
    
    result = await db.execute(
        select(Payment).where(
            Payment.id == payment_id,
            Payment.user_id == current_user.id
        )
    )
    payment = result.scalar_one_or_none()
    
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found"
        )
    
    return PaymentResponse.from_orm(payment)


@router.get(
    "/",
    response_model=PaymentList,
    summary="List User Payments",
    description="List all payments for the authenticated user."
)
async def list_payments(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentList:
    """
    List user payments with pagination.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Paginated list of payments
    """
    from sqlalchemy import select, func
    from app.payments.models.payment import Payment
    
    # Get total count
    count_result = await db.execute(
        select(func.count(Payment.id)).where(Payment.user_id == current_user.id)
    )
    total_count = count_result.scalar()
    
    # Get paginated payments
    offset = (page - 1) * page_size
    result = await db.execute(
        select(Payment)
        .where(Payment.user_id == current_user.id)
        .order_by(Payment.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    payments = result.scalars().all()
    
    return PaymentList(
        payments=[PaymentResponse.from_orm(p) for p in payments],
        total_count=total_count,
        page=page,
        page_size=page_size
    )


@router.get(
    "/analytics/summary",
    response_model=PaymentAnalytics,
    summary="Get Payment Analytics",
    description="Get payment analytics for the authenticated user."
)
async def get_payment_analytics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PaymentAnalytics:
    """
    Get payment analytics summary.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Payment analytics data
    """
    from sqlalchemy import select, func
    import sqlalchemy as sa
    from app.payments.models.payment import Payment, PaymentStatus
    from datetime import datetime, timedelta
    
    # Get data for last 30 days
    period_start = datetime.utcnow() - timedelta(days=30)
    period_end = datetime.utcnow()
    
    # Total revenue
    revenue_result = await db.execute(
        select(func.sum(Payment.amount))
        .where(
            Payment.user_id == current_user.id,
            Payment.status == PaymentStatus.COMPLETED,
            Payment.created_at >= period_start
        )
    )
    total_revenue = float(revenue_result.scalar() or 0)
    
    # Transaction counts
    counts_result = await db.execute(
        select(
            func.count(Payment.id).label("total"),
            func.sum(func.cast(Payment.status == PaymentStatus.COMPLETED, sa.Integer)).label("successful"),
            func.sum(func.cast(Payment.status == PaymentStatus.FAILED, sa.Integer)).label("failed"),
            func.sum(func.cast(Payment.status == PaymentStatus.REFUNDED, sa.Integer)).label("refunded"),
        )
        .where(
            Payment.user_id == current_user.id,
            Payment.created_at >= period_start
        )
    )
    counts = counts_result.one()
    
    # Fees
    fees_result = await db.execute(
        select(
            func.sum(Payment.platform_fee).label("platform_fees"),
            func.sum(Payment.processing_fee).label("stripe_fees")
        )
        .where(
            Payment.user_id == current_user.id,
            Payment.status == PaymentStatus.COMPLETED,
            Payment.created_at >= period_start
        )
    )
    fees = fees_result.one()
    
    return PaymentAnalytics(
        total_revenue=total_revenue,
        total_transactions=counts.total,
        successful_transactions=counts.successful or 0,
        failed_transactions=counts.failed or 0,
        refunded_transactions=counts.refunded or 0,
        average_transaction_value=total_revenue / counts.total if counts.total > 0 else 0,
        total_platform_fees=float(fees.platform_fees or 0),
        total_stripe_fees=float(fees.stripe_fees or 0),
        net_revenue=total_revenue - float(fees.platform_fees or 0) - float(fees.stripe_fees or 0),
        period_start=period_start,
        period_end=period_end
    )
