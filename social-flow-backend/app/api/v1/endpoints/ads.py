"""Ad endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.user import User
from app.api.v1.endpoints.auth import get_current_active_user

router = APIRouter()

@router.get("/")
async def get_ads(db: AsyncSession = Depends(get_db)) -> list:
    """Get ads for user."""
    return []

@router.post("/{ad_id}/impression")
async def record_impression(ad_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Record ad impression."""
    return {"message": f"Record impression for ad {ad_id} - TODO"}

@router.post("/{ad_id}/click")
async def record_click(ad_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Record ad click."""
    return {"message": f"Record click for ad {ad_id} - TODO"}
