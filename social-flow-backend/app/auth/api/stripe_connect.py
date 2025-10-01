"""
Stripe Connect API Routes.

Creator payout accounts and payout management.
"""

from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.auth.models.user import User
from app.payments.schemas.payment import (
    ConnectAccountCreate,
    ConnectAccountResponse,
    ConnectAccountStatus,
    PayoutCreate,
    PayoutResponse,
    PayoutList,
    RevenueBreakdown,
    CreatorEarnings,
)
from app.payments.services.stripe_payment_service import StripePaymentService


router = APIRouter(prefix="/connect", tags=["Creator Payouts"])


@router.post(
    "/account",
    response_model=ConnectAccountResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Connect Account",
    description="Create a Stripe Connect account for receiving payouts."
)
async def create_connect_account(
    request: ConnectAccountCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ConnectAccountResponse:
    """
    Create Stripe Connect account for creator payouts.
    
    Args:
        request: Connect account creation request
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Connect account details with onboarding URL
    """
    service = StripePaymentService(db)
    
    try:
        account, onboarding_url = await service.create_connect_account(
            user_id=current_user.id,
            country=request.country,
            account_type=request.account_type
        )
        
        response = ConnectAccountResponse.from_orm(account)
        response.onboarding_url = onboarding_url
        
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create Connect account: {str(e)}"
        )


@router.get(
    "/account",
    response_model=ConnectAccountStatus,
    summary="Get Connect Account Status",
    description="Get the authenticated user's Connect account status."
)
async def get_connect_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ConnectAccountStatus:
    """
    Get Connect account status and requirements.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Connect account status with onboarding requirements
    """
    service = StripePaymentService(db)
    
    try:
        account = await service.get_connect_account_status(user_id=current_user.id)
        
        return ConnectAccountStatus(
            connect_account_id=str(account.id),
            stripe_account_id=account.stripe_account_id,
            status=account.status,
            charges_enabled=account.charges_enabled,
            payouts_enabled=account.payouts_enabled,
            details_submitted=account.details_submitted,
            is_fully_onboarded=account.is_fully_onboarded,
            requirements_currently_due=account.requirements_currently_due or [],
            available_balance=float(account.available_balance),
            pending_balance=float(account.pending_balance)
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get Connect account: {str(e)}"
        )


@router.post(
    "/payouts",
    response_model=PayoutResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Payout",
    description="Create a payout for the creator (admin only)."
)
async def create_payout(
    request: PayoutCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PayoutResponse:
    """
    Create payout for creator.
    
    Note: This endpoint should be restricted to admin users or automated by cron jobs.
    
    Args:
        request: Payout creation request
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Created payout details
    """
    service = StripePaymentService(db)
    
    try:
        payout = await service.create_payout(
            user_id=current_user.id,
            period_start=request.period_start,
            period_end=request.period_end,
            revenue_breakdown=request.revenue_breakdown
        )
        
        return PayoutResponse.from_orm(payout)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create payout: {str(e)}"
        )


@router.get(
    "/payouts",
    response_model=PayoutList,
    summary="List Payouts",
    description="List all payouts for the authenticated creator."
)
async def list_payouts(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PayoutList:
    """
    List creator payouts with pagination.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Paginated list of payouts
    """
    from sqlalchemy import select, func
    from app.payments.models.stripe_connect import CreatorPayout
    
    # Get total count
    count_result = await db.execute(
        select(func.count(CreatorPayout.id)).where(CreatorPayout.user_id == current_user.id)
    )
    total_count = count_result.scalar()
    
    # Get total amount
    amount_result = await db.execute(
        select(func.sum(CreatorPayout.net_amount))
        .where(
            CreatorPayout.user_id == current_user.id,
            CreatorPayout.status == "paid"
        )
    )
    total_amount = float(amount_result.scalar() or 0)
    
    # Get paginated payouts
    offset = (page - 1) * page_size
    result = await db.execute(
        select(CreatorPayout)
        .where(CreatorPayout.user_id == current_user.id)
        .order_by(CreatorPayout.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    payouts = result.scalars().all()
    
    return PayoutList(
        payouts=[PayoutResponse.from_orm(p) for p in payouts],
        total_count=total_count,
        total_amount=total_amount,
        page=page,
        page_size=page_size
    )


@router.get(
    "/payouts/{payout_id}",
    response_model=PayoutResponse,
    summary="Get Payout Details",
    description="Get details for a specific payout."
)
async def get_payout(
    payout_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PayoutResponse:
    """
    Get payout details.
    
    Args:
        payout_id: Payout ID
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Payout details
    """
    from sqlalchemy import select
    from app.payments.models.stripe_connect import CreatorPayout
    
    result = await db.execute(
        select(CreatorPayout).where(
            CreatorPayout.id == payout_id,
            CreatorPayout.user_id == current_user.id
        )
    )
    payout = result.scalar_one_or_none()
    
    if not payout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payout not found"
        )
    
    return PayoutResponse.from_orm(payout)


@router.get(
    "/revenue/breakdown",
    response_model=RevenueBreakdown,
    summary="Get Revenue Breakdown",
    description="Get revenue breakdown for a specific period."
)
async def get_revenue_breakdown(
    period_start: datetime,
    period_end: datetime,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> RevenueBreakdown:
    """
    Get revenue breakdown by source for a period.
    
    Args:
        period_start: Period start date
        period_end: Period end date
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Revenue breakdown by source
    """
    from sqlalchemy import select, func
    from app.payments.models.stripe_connect import CreatorPayout
    
    # Get revenue breakdown
    result = await db.execute(
        select(
            func.sum(CreatorPayout.subscription_revenue).label("subscription"),
            func.sum(CreatorPayout.tips_revenue).label("tips"),
            func.sum(CreatorPayout.content_sales_revenue).label("content_sales"),
            func.sum(CreatorPayout.ad_revenue).label("ad_revenue"),
            func.sum(CreatorPayout.gross_amount).label("total_revenue"),
            func.sum(CreatorPayout.platform_fee).label("platform_fee"),
            func.sum(CreatorPayout.stripe_fee).label("stripe_fee"),
            func.sum(CreatorPayout.net_amount).label("net_revenue"),
        )
        .where(
            CreatorPayout.user_id == current_user.id,
            CreatorPayout.period_start >= period_start,
            CreatorPayout.period_end <= period_end
        )
    )
    breakdown = result.one()
    
    return RevenueBreakdown(
        period_start=period_start,
        period_end=period_end,
        subscription_revenue=float(breakdown.subscription or 0),
        tips_revenue=float(breakdown.tips or 0),
        content_sales_revenue=float(breakdown.content_sales or 0),
        ad_revenue=float(breakdown.ad_revenue or 0),
        total_revenue=float(breakdown.total_revenue or 0),
        platform_fee=float(breakdown.platform_fee or 0),
        stripe_fee=float(breakdown.stripe_fee or 0),
        net_revenue=float(breakdown.net_revenue or 0)
    )


@router.get(
    "/earnings/summary",
    response_model=CreatorEarnings,
    summary="Get Creator Earnings",
    description="Get earnings summary for the authenticated creator."
)
async def get_creator_earnings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> CreatorEarnings:
    """
    Get creator earnings summary.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Creator earnings summary
    """
    from sqlalchemy import select, func
    from app.payments.models.stripe_connect import CreatorPayout
    
    # Total earnings
    total_result = await db.execute(
        select(func.sum(CreatorPayout.net_amount))
        .where(CreatorPayout.user_id == current_user.id)
    )
    total_earnings = float(total_result.scalar() or 0)
    
    # Pending earnings
    pending_result = await db.execute(
        select(func.sum(CreatorPayout.net_amount))
        .where(
            CreatorPayout.user_id == current_user.id,
            CreatorPayout.status.in_(["pending", "processing"])
        )
    )
    pending_earnings = float(pending_result.scalar() or 0)
    
    # Paid earnings
    paid_result = await db.execute(
        select(func.sum(CreatorPayout.net_amount))
        .where(
            CreatorPayout.user_id == current_user.id,
            CreatorPayout.status == "paid"
        )
    )
    paid_earnings = float(paid_result.scalar() or 0)
    
    # Revenue by source
    source_result = await db.execute(
        select(
            func.sum(CreatorPayout.subscription_revenue).label("subscription"),
            func.sum(CreatorPayout.tips_revenue).label("tips"),
            func.sum(CreatorPayout.content_sales_revenue).label("content_sales"),
            func.sum(CreatorPayout.ad_revenue).label("ad_revenue"),
        )
        .where(CreatorPayout.user_id == current_user.id)
    )
    source_breakdown = source_result.one()
    
    # Fees
    fees_result = await db.execute(
        select(
            func.sum(CreatorPayout.platform_fee).label("platform_fees"),
            func.sum(CreatorPayout.stripe_fee).label("stripe_fees"),
        )
        .where(CreatorPayout.user_id == current_user.id)
    )
    fees = fees_result.one()
    
    # Next payout (if any)
    next_payout_result = await db.execute(
        select(CreatorPayout)
        .where(
            CreatorPayout.user_id == current_user.id,
            CreatorPayout.status == "pending",
            CreatorPayout.scheduled_at.isnot(None)
        )
        .order_by(CreatorPayout.scheduled_at)
        .limit(1)
    )
    next_payout = next_payout_result.scalar_one_or_none()
    
    return CreatorEarnings(
        user_id=str(current_user.id),
        total_earnings=total_earnings,
        pending_earnings=pending_earnings,
        paid_earnings=paid_earnings,
        subscription_earnings=float(source_breakdown.subscription or 0),
        tips_earnings=float(source_breakdown.tips or 0),
        content_sales_earnings=float(source_breakdown.content_sales or 0),
        ad_revenue_earnings=float(source_breakdown.ad_revenue or 0),
        platform_fees_paid=float(fees.platform_fees or 0),
        stripe_fees_paid=float(fees.stripe_fees or 0),
        next_payout_date=next_payout.scheduled_at if next_payout else None,
        next_payout_amount=float(next_payout.net_amount) if next_payout else 0.0,
        currency="usd"
    )
