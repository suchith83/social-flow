"""Comment API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/")
async def create_comment(comment_data: dict, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Create a comment."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.get("/{comment_id}")
async def get_comment(comment_id: str, db: AsyncSession = Depends(get_db)):
    """Get comment by ID."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.delete("/{comment_id}")
async def delete_comment(comment_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Delete comment."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

