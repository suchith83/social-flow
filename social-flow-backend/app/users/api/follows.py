"""Follow/unfollow API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/{user_id}")
async def follow_user(user_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Follow a user."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.delete("/{user_id}")
async def unfollow_user(user_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Unfollow a user."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.get("/{user_id}/followers")
async def get_followers(user_id: str, db: AsyncSession = Depends(get_db)):
    """Get user followers."""
    return []

@router.get("/{user_id}/following")
async def get_following(user_id: str, db: AsyncSession = Depends(get_db)):
    """Get users that user is following."""
    return []

