"""Like API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/{post_id}")
async def like_post(post_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Like a post."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.delete("/{post_id}")
async def unlike_post(post_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Unlike a post."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.get("/{post_id}")
async def get_post_likes(post_id: str, db: AsyncSession = Depends(get_db)):
    """Get users who liked a post."""
    return []

