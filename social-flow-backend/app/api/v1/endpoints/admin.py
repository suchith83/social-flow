"""Admin endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.auth.models.user import User
from app.auth.api.auth import get_current_active_user

router = APIRouter()

@router.get("/stats")
async def get_admin_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get admin statistics."""
    return {"message": "Get admin stats - TODO"}

@router.get("/health")
async def get_system_health(db: AsyncSession = Depends(get_db)) -> dict:
    """Get system health."""
    return {"status": "healthy"}
