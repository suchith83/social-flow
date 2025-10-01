"""Follow endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.auth.models.user import User
from app.auth.api.auth import get_current_active_user

router = APIRouter()

@router.post("/{user_id}")
async def follow_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Follow a user."""
    return {"message": f"Follow user {user_id} - TODO"}

@router.delete("/{user_id}")
async def unfollow_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Unfollow a user."""
    return {"message": f"Unfollow user {user_id} - TODO"}
