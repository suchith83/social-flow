"""Advertisement API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/")
async def create_ad(ad_data: dict, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Create an advertisement."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.get("/{ad_id}")
async def get_ad(ad_id: str, db: AsyncSession = Depends(get_db)):
    """Get ad by ID."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

@router.get("/")
async def list_ads(skip: int = 0, limit: int = 20, db: AsyncSession = Depends(get_db)):
    """List ads."""
    return []

@router.put("/{ad_id}")
async def update_ad(ad_id: str, ad_data: dict, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Update ad."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")

