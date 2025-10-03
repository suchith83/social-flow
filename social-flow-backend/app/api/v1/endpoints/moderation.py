"""Moderation endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.auth.api.auth import get_current_active_user

router = APIRouter()

@router.get("/flagged")
async def get_flagged_content(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """Get flagged content."""
    return []

@router.post("/moderate")
async def moderate_content(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Moderate content."""
    return {"message": "Moderate content - TODO"}

