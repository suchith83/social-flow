"""Subscription endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.auth.models.user import User
from app.auth.api.auth import get_current_active_user

router = APIRouter()

@router.post("/")
async def create_subscription(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a subscription."""
    return {"message": "Create subscription - TODO"}

@router.get("/")
async def get_subscriptions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """Get user subscriptions."""
    return []
