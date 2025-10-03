"""User API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return {"id": str(current_user.id), "username": current_user.username, "email": current_user.email}

@router.get("/{user_id}")
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    """Get user by ID."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.put("/{user_id}")
async def update_user(user_id: str, user_data: dict, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Update user."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

