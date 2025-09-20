"""Comment endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_active_user

router = APIRouter()

@router.post("/")
async def create_comment(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new comment."""
    return {"message": "Create comment - TODO"}

@router.get("/{comment_id}")
async def get_comment(comment_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Get comment by ID."""
    return {"message": f"Get comment {comment_id} - TODO"}
