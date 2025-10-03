"""
Notification endpoints.

This module contains all notification-related API endpoints.
"""

from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import NotificationServiceError
from app.models.user import User
from app.auth.api.auth import get_current_active_user
from app.notifications.services.notification_service import notification_service

router = APIRouter()


@router.get("/")
async def get_notifications(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    notification_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get user notifications."""
    try:
        notifications = await notification_service.get_user_notifications(
            str(current_user.id), limit, offset, notification_type
        )
        return {"notifications": notifications}
    except NotificationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get notifications")


@router.post("/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Mark notification as read."""
    try:
        result = await notification_service.mark_notification_read(
            notification_id, str(current_user.id)
        )
        return result
    except NotificationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")


@router.post("/read-all")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Mark all notifications as read."""
    try:
        result = await notification_service.mark_all_notifications_read(str(current_user.id))
        return result
    except NotificationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to mark all notifications as read")


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Delete a notification."""
    try:
        result = await notification_service.delete_notification(
            notification_id, str(current_user.id)
        )
        return result
    except NotificationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete notification")


@router.get("/preferences")
async def get_notification_preferences(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get notification preferences."""
    try:
        preferences = await notification_service.get_notification_preferences(str(current_user.id))
        return preferences
    except NotificationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get notification preferences")


@router.put("/preferences")
async def update_notification_preferences(
    preferences: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Update notification preferences."""
    try:
        result = await notification_service.update_notification_preferences(
            str(current_user.id), preferences
        )
        return result
    except NotificationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update notification preferences")


@router.get("/stats")
async def get_notification_stats(
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get notification statistics."""
    try:
        stats = await notification_service.get_notification_stats(
            str(current_user.id), time_range
        )
        return stats
    except NotificationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get notification stats")

