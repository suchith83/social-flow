"""
Ads endpoints.

This module contains all advertisement-related API endpoints.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import AdsServiceError
from app.auth.models.user import User
from app.auth.api.auth import get_current_active_user
from app.ads.services.ads_service import ads_service

router = APIRouter()


@router.get("/video/{video_id}")
async def get_ads_for_video(
    video_id: str,
    ad_type: str = Query("pre-roll"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get ads for a video."""
    try:
        ads = await ads_service.get_ads_for_video(video_id, str(current_user.id), ad_type)
        return {"ads": ads}
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get ads")


@router.post("/impression")
async def track_impression(
    ad_id: str,
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Track ad impression."""
    try:
        result = await ads_service.track_ad_impression(ad_id, str(current_user.id), video_id)
        return result
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to track impression")


@router.post("/click")
async def track_click(
    ad_id: str,
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Track ad click."""
    try:
        result = await ads_service.track_ad_click(ad_id, str(current_user.id), video_id)
        return result
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to track click")


@router.get("/analytics/{ad_id}")
async def get_ad_analytics(
    ad_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get analytics for a specific ad."""
    try:
        result = await ads_service.get_ad_analytics(ad_id, time_range)
        return result
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get ad analytics")


@router.post("/campaigns")
async def create_ad_campaign(
    campaign_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a new ad campaign."""
    try:
        result = await ads_service.create_ad_campaign(campaign_data)
        return result
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create ad campaign")


@router.get("/campaigns")
async def get_ad_campaigns(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get ad campaigns for a user."""
    try:
        campaigns = await ads_service.get_ad_campaigns(str(current_user.id), limit)
        return {"campaigns": campaigns}
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get ad campaigns")


@router.put("/campaigns/{campaign_id}")
async def update_ad_campaign(
    campaign_id: str,
    updates: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update an ad campaign."""
    try:
        result = await ads_service.update_ad_campaign(campaign_id, updates)
        return result
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update ad campaign")


@router.delete("/campaigns/{campaign_id}")
async def delete_ad_campaign(
    campaign_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Delete an ad campaign."""
    try:
        result = await ads_service.delete_ad_campaign(campaign_id)
        return result
    except AdsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete ad campaign")
