"""
Notification API Routes

REST API endpoints for notification management:
- Get user notifications
- Mark as read/unread
- Delete notifications
- Manage preferences
- Register push tokens
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from app.core.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.notifications.services.enhanced_service import EnhancedNotificationService
from app.notifications.models.notification import Notification

router = APIRouter(prefix="/notifications", tags=["notifications"])


# Request/Response Models

class NotificationResponse(BaseModel):
    """Notification response model"""
    id: str
    type: str
    title: str
    message: str
    status: str
    priority: str = "normal"
    action_url: Optional[str] = None
    action_text: Optional[str] = None
    icon: Optional[str] = None
    image_url: Optional[str] = None
    is_read: bool
    created_at: datetime
    read_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class NotificationListResponse(BaseModel):
    """List of notifications with count"""
    notifications: List[NotificationResponse]
    total: int
    unread_count: int


class NotificationPreferencesRequest(BaseModel):
    """Notification preferences update request"""
    email_enabled: Optional[bool] = None
    push_enabled: Optional[bool] = None
    sms_enabled: Optional[bool] = None
    new_follower_enabled: Optional[bool] = None
    new_like_enabled: Optional[bool] = None
    new_comment_enabled: Optional[bool] = None
    mention_enabled: Optional[bool] = None
    video_processing_enabled: Optional[bool] = None
    live_stream_enabled: Optional[bool] = None
    moderation_enabled: Optional[bool] = None
    copyright_enabled: Optional[bool] = None
    payment_enabled: Optional[bool] = None
    payout_enabled: Optional[bool] = None
    donation_enabled: Optional[bool] = None
    system_enabled: Optional[bool] = None
    security_enabled: Optional[bool] = None
    daily_digest_enabled: Optional[bool] = None
    weekly_digest_enabled: Optional[bool] = None
    digest_time: Optional[str] = None
    quiet_hours_enabled: Optional[bool] = None
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None


class NotificationPreferencesResponse(BaseModel):
    """Notification preferences response"""
    email_enabled: bool
    push_enabled: bool
    sms_enabled: bool
    new_follower_enabled: bool
    new_like_enabled: bool
    new_comment_enabled: bool
    mention_enabled: bool
    video_processing_enabled: bool
    live_stream_enabled: bool
    moderation_enabled: bool
    copyright_enabled: bool
    payment_enabled: bool
    payout_enabled: bool
    donation_enabled: bool
    system_enabled: bool
    security_enabled: bool
    daily_digest_enabled: bool
    weekly_digest_enabled: bool
    digest_time: str
    quiet_hours_enabled: bool
    quiet_hours_start: Optional[str]
    quiet_hours_end: Optional[str]
    
    class Config:
        from_attributes = True


class PushTokenRequest(BaseModel):
    """Push token registration request"""
    token: str = Field(..., description="FCM/APNS token")
    device_id: str = Field(..., description="Unique device identifier")
    device_type: str = Field(..., description="Device type (ios/android)")
    device_name: Optional[str] = Field(None, description="Device name")


class PushTokenResponse(BaseModel):
    """Push token response"""
    id: str
    device_type: str
    device_name: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# Routes

@router.get("/", response_model=NotificationListResponse)
async def get_notifications(
    limit: int = Query(50, ge=1, le=100, description="Number of notifications to return"),
    offset: int = Query(0, ge=0, description="Number of notifications to skip"),
    unread_only: bool = Query(False, description="Return only unread notifications"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user notifications.
    
    Returns paginated list of notifications with unread count.
    """
    
    service = EnhancedNotificationService(db)
    
    # Get notifications
    notifications = await service.get_user_notifications(
        user_id=str(current_user.id),
        limit=limit,
        offset=offset,
        unread_only=unread_only
    )
    
    # Get unread count
    unread_count = await service.get_unread_count(str(current_user.id))
    
    return NotificationListResponse(
        notifications=[NotificationResponse.from_orm(n) for n in notifications],
        total=len(notifications),
        unread_count=unread_count
    )


@router.get("/unread-count")
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get count of unread notifications.
    
    Returns integer count of unread notifications.
    """
    
    service = EnhancedNotificationService(db)
    count = await service.get_unread_count(str(current_user.id))
    
    return {"unread_count": count}


@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Mark notification as read.
    
    Sets notification status to read and records read timestamp.
    """
    
    service = EnhancedNotificationService(db)
    notification = await service.mark_as_read(
        notification_id=notification_id,
        user_id=str(current_user.id)
    )
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return {"status": "success", "notification_id": notification_id}


@router.post("/mark-all-read")
async def mark_all_as_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Mark all notifications as read.
    
    Updates all unread notifications to read status.
    """
    
    service = EnhancedNotificationService(db)
    count = await service.mark_all_as_read(str(current_user.id))
    
    return {"status": "success", "marked_count": count}


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete notification.
    
    Permanently removes notification from database.
    """
    
    service = EnhancedNotificationService(db)
    success = await service.delete_notification(
        notification_id=notification_id,
        user_id=str(current_user.id)
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return {"status": "success", "notification_id": notification_id}


@router.get("/preferences", response_model=NotificationPreferencesResponse)
async def get_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get notification preferences.
    
    Returns user's notification settings for all channels and types.
    """
    
    service = EnhancedNotificationService(db)
    preferences = await service.get_preferences(str(current_user.id))
    
    if not preferences:
        # Return default preferences
        return NotificationPreferencesResponse(
            email_enabled=True,
            push_enabled=True,
            sms_enabled=False,
            new_follower_enabled=True,
            new_like_enabled=True,
            new_comment_enabled=True,
            mention_enabled=True,
            video_processing_enabled=True,
            live_stream_enabled=True,
            moderation_enabled=True,
            copyright_enabled=True,
            payment_enabled=True,
            payout_enabled=True,
            donation_enabled=True,
            system_enabled=True,
            security_enabled=True,
            daily_digest_enabled=False,
            weekly_digest_enabled=True,
            digest_time="09:00",
            quiet_hours_enabled=False,
            quiet_hours_start=None,
            quiet_hours_end=None
        )
    
    return NotificationPreferencesResponse.from_orm(preferences)


@router.put("/preferences", response_model=NotificationPreferencesResponse)
async def update_preferences(
    request: NotificationPreferencesRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update notification preferences.
    
    Updates user's notification settings. Only provided fields are updated.
    """
    
    service = EnhancedNotificationService(db)
    
    # Get non-None values
    updates = {k: v for k, v in request.dict().items() if v is not None}
    
    preferences = await service.create_or_update_preferences(
        user_id=str(current_user.id),
        preferences=updates
    )
    
    return NotificationPreferencesResponse.from_orm(preferences)


@router.post("/push-tokens", response_model=PushTokenResponse)
async def register_push_token(
    request: PushTokenRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Register push notification token.
    
    Registers device token for push notifications (FCM/APNS).
    """
    
    service = EnhancedNotificationService(db)
    
    token = await service.register_push_token(
        user_id=str(current_user.id),
        token=request.token,
        device_id=request.device_id,
        device_type=request.device_type,
        device_name=request.device_name
    )
    
    return PushTokenResponse.from_orm(token)


@router.delete("/push-tokens/{token}")
async def unregister_push_token(
    token: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Unregister push notification token.
    
    Deactivates device token to stop push notifications.
    """
    
    service = EnhancedNotificationService(db)
    success = await service.unregister_push_token(token)
    
    if not success:
        raise HTTPException(status_code=404, detail="Token not found")
    
    return {"status": "success", "token": token}

