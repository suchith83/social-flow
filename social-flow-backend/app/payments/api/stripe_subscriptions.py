"""Stripe Subscriptions API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/")
async def create_subscription(subscription_data: dict, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Create Stripe subscription."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.delete("/{subscription_id}")
async def cancel_subscription(subscription_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Cancel Stripe subscription."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

