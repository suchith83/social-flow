"""Search endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_active_user

router = APIRouter()

@router.get("/")
async def search(
    q: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """Search content."""
    return []

@router.get("/suggestions")
async def get_suggestions(
    q: str,
    db: AsyncSession = Depends(get_db),
) -> list:
    """Get search suggestions."""
    return []
