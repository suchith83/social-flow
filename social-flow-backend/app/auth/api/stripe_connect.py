"""
Stripe Connect API endpoints.

This module defines API endpoints for Stripe Connect integration.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/stripe-connect", tags=["stripe-connect"])


@router.post("/onboard", response_model=dict)
async def onboard_stripe_connect(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Onboard user to Stripe Connect."""
    # Placeholder implementation
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Stripe Connect onboarding not yet implemented"
    )


@router.get("/account", response_model=dict)
async def get_stripe_connect_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get Stripe Connect account details."""
    # Placeholder implementation
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Stripe Connect account retrieval not yet implemented"
    )


@router.delete("/account", response_model=dict)
async def disconnect_stripe_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Disconnect Stripe Connect account."""
    # Placeholder implementation
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Stripe Connect disconnection not yet implemented"
    )

