"""Stripe Payments API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/payment-intent")
async def create_payment_intent(amount: int, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Create Stripe payment intent."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.post("/checkout-session")
async def create_checkout_session(price_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Create Stripe checkout session."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

