"""Like endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_active_user

router = APIRouter()

@router.post("/")
async def create_like(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new like."""
    return {"message": "Create like - TODO"}

@router.delete("/{like_id}")
async def delete_like(like_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Delete like by ID."""
    return {"message": f"Delete like {like_id} - TODO"}
