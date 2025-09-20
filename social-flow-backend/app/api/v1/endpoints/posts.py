"""Post endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_active_user

router = APIRouter()

@router.post("/")
async def create_post(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new post."""
    return {"message": "Create post - TODO"}

@router.get("/{post_id}")
async def get_post(post_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Get post by ID."""
    return {"message": f"Get post {post_id} - TODO"}

@router.get("/")
async def get_posts(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)) -> list:
    """Get list of posts."""
    return []
