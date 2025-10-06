"""
Notification endpoints for the Social Flow API.

This module provides comprehensive notification management including
in-app notifications, preferences, and push notification tokens.
"""

from typing import Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_db,
    get_current_user,
)
from app.models.user import User
from app.models.notification import NotificationType
from app.infrastructure.crud import crud_notification

router = APIRouter()


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================


class NotificationResponse(BaseModel):
    """Notification response schema."""
    id: str
    notification_type: str
    title: str
    message: str
    data: Optional[dict] = None
    is_read: bool
    read_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class NotificationList(BaseModel):
    """Notification list response."""
    notifications: list[NotificationResponse]
    total_count: int
    unread_count: int
    page: int
    page_size: int


class NotificationSettingsResponse(BaseModel):
    """Notification settings response."""
    email_enabled: bool
    push_enabled: bool
    in_app_enabled: bool
    likes_enabled: bool
    comments_enabled: bool
    follows_enabled: bool
    mentions_enabled: bool
    live_streams_enabled: bool
    subscriptions_enabled: bool
    payments_enabled: bool
    moderation_enabled: bool
    system_enabled: bool
    
    class Config:
        from_attributes = True


class NotificationSettingsUpdate(BaseModel):
    """Update notification settings request."""
    email_enabled: Optional[bool] = None
    push_enabled: Optional[bool] = None
    in_app_enabled: Optional[bool] = None
    likes_enabled: Optional[bool] = None
    comments_enabled: Optional[bool] = None
    follows_enabled: Optional[bool] = None
    mentions_enabled: Optional[bool] = None
    live_streams_enabled: Optional[bool] = None
    subscriptions_enabled: Optional[bool] = None
    payments_enabled: Optional[bool] = None
    moderation_enabled: Optional[bool] = None
    system_enabled: Optional[bool] = None


class PushTokenCreate(BaseModel):
    """Register push token request."""
    token: str = Field(..., description="FCM or APNS token")
    device_type: str = Field(..., description="Device type: ios, android, web")
    device_name: Optional[str] = Field(None, description="Device name for identification")


class PushTokenResponse(BaseModel):
    """Push token response."""
    id: str
    token: str
    device_type: str
    device_name: Optional[str]
    created_at: datetime
    last_used_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class UnreadCountResponse(BaseModel):
    """Unread notification count response."""
    unread_count: int


class MarkReadRequest(BaseModel):
    """Mark notifications as read request."""
    notification_ids: list[str] = Field(..., description="List of notification IDs to mark as read")


# ============================================================================
# PUSH TOKEN ENDPOINTS (Placeholders)
# ============================================================================



# ============================================================================
# NOTIFICATION ENDPOINTS
# ============================================================================


@router.get(
    "/notifications",
    response_model=NotificationList,
    summary="List notifications",
    description="Get user's notifications with pagination and filtering",
)
async def list_notifications(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    unread_only: bool = Query(False, description="Show only unread notifications"),
    notification_type: Optional[str] = Query(None, description="Filter by notification type"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationList:
    """
    List user's notifications with filters.
    
    **Features:**
    - Paginated results
    - Filter by unread status
    - Filter by notification type
    - Sorted by newest first
    
    **Query Parameters:**
    - skip: Pagination offset (default: 0)
    - limit: Results per page (default: 20, max: 100)
    - unread_only: Show only unread (default: false)
    - notification_type: Filter by type (optional)
    """
    # Parse notification type filter
    type_filter = None
    if notification_type:
        try:
            type_filter = NotificationType(notification_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid notification type: {notification_type}",
            )
    
    # Get notifications
    notifications = await crud_notification.notification.get_by_user(
        db,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        unread_only=unread_only,
        notification_type=type_filter,
    )
    
    # Get unread count
    unread_count = await crud_notification.notification.get_unread_count(
        db, user_id=current_user.id
    )
    
    return NotificationList(
        notifications=[
            NotificationResponse(
                id=str(n.id),
                notification_type=n.notification_type.value,
                title=n.title,
                message=n.message,
                data=n.data,
                is_read=n.is_read,
                read_at=n.read_at,
                created_at=n.created_at,
            )
            for n in notifications
        ],
        total_count=len(notifications) + skip,
        unread_count=unread_count,
        page=skip // limit + 1,
        page_size=limit,
    )


@router.get(
    "/notifications/unread-count",
    response_model=UnreadCountResponse,
    summary="Get unread count",
    description="Get count of unread notifications",
)
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UnreadCountResponse:
    """
    Get unread notification count.
    
    Lightweight endpoint for polling unread count.
    Useful for displaying notification badge.
    """
    unread_count = await crud_notification.notification.get_unread_count(
        db, user_id=current_user.id
    )
    
    return UnreadCountResponse(unread_count=unread_count)


@router.get(
    "/notifications/{notification_id}",
    response_model=NotificationResponse,
    summary="Get notification",
    description="Get specific notification details",
)
async def get_notification(
    notification_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationResponse:
    """
    Get notification details.
    
    Returns full notification information including data payload.
    """
    notification = await crud_notification.notification.get(db, notification_id)
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found",
        )
    
    # Verify ownership
    if notification.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this notification",
        )
    
    return NotificationResponse(
        id=str(notification.id),
        notification_type=notification.notification_type.value,
        title=notification.title,
        message=notification.message,
        data=notification.data,
        is_read=notification.is_read,
        read_at=notification.read_at,
        created_at=notification.created_at,
    )


@router.post(
    "/notifications/{notification_id}/read",
    response_model=NotificationResponse,
    summary="Mark notification as read",
    description="Mark a single notification as read",
)
async def mark_notification_read(
    notification_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationResponse:
    """
    Mark notification as read.
    
    Updates is_read flag and sets read_at timestamp.
    """
    notification = await crud_notification.notification.get(db, notification_id)
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found",
        )
    
    # Verify ownership
    if notification.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify this notification",
        )
    
    # Mark as read
    notification = await crud_notification.notification.mark_as_read(
        db, notification_id=notification_id
    )
    
    return NotificationResponse(
        id=str(notification.id),
        notification_type=notification.notification_type.value,
        title=notification.title,
        message=notification.message,
        data=notification.data,
        is_read=notification.is_read,
        read_at=notification.read_at,
        created_at=notification.created_at,
    )


@router.post(
    "/notifications/mark-all-read",
    summary="Mark all as read",
    description="Mark all notifications as read",
)
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Mark all notifications as read.
    
    Bulk operation to mark all user's notifications as read.
    Returns count of notifications marked.
    """
    count = await crud_notification.notification.mark_all_as_read(
        db, user_id=current_user.id
    )
    
    return {
        "success": True,
        "message": f"Marked {count} notifications as read",
        "count": count,
    }


@router.delete(
    "/notifications/{notification_id}",
    summary="Delete notification",
    description="Delete a notification",
)
async def delete_notification(
    notification_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Delete a notification.
    
    Permanently removes notification from user's list.
    """
    notification = await crud_notification.notification.get(db, notification_id)
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found",
        )
    
    # Verify ownership
    if notification.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this notification",
        )
    
    await crud_notification.notification.delete(db, notification_id)
    
    return {
        "success": True,
        "message": "Notification deleted",
    }


# ============================================================================
# NOTIFICATION SETTINGS ENDPOINTS
# ============================================================================


@router.get(
    "/notifications/settings",
    response_model=NotificationSettingsResponse,
    summary="Get notification settings",
    description="Get user's notification preferences",
)
async def get_notification_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationSettingsResponse:
    """
    Get notification settings.
    
    Returns user's notification preferences for all channels
    and notification types. Creates default settings if none exist.
    """
    settings = await crud_notification.notification_settings.get_or_create(
        db, user_id=current_user.id
    )
    
    return NotificationSettingsResponse(
        email_enabled=settings.email_enabled,
        push_enabled=settings.push_enabled,
        in_app_enabled=settings.in_app_enabled,
        likes_enabled=settings.likes_enabled,
        comments_enabled=settings.comments_enabled,
        follows_enabled=settings.follows_enabled,
        mentions_enabled=settings.mentions_enabled,
        live_streams_enabled=settings.live_streams_enabled,
        subscriptions_enabled=settings.subscriptions_enabled,
        payments_enabled=settings.payments_enabled,
        moderation_enabled=settings.moderation_enabled,
        system_enabled=settings.system_enabled,
    )


@router.put(
    "/notifications/settings",
    response_model=NotificationSettingsResponse,
    summary="Update notification settings",
    description="Update user's notification preferences",
)
async def update_notification_settings(
    settings_update: NotificationSettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationSettingsResponse:
    """
    Update notification settings.
    
    Updates user's notification preferences for channels and types.
    Only provided fields are updated, others remain unchanged.
    
    **Channel Settings:**
    - email_enabled: Email notifications
    - push_enabled: Push notifications
    - in_app_enabled: In-app notifications
    
    **Type Settings:**
    - likes_enabled: Like notifications
    - comments_enabled: Comment notifications
    - follows_enabled: Follow notifications
    - mentions_enabled: Mention notifications
    - live_streams_enabled: Live stream notifications
    - subscriptions_enabled: Subscription notifications
    - payments_enabled: Payment notifications
    - moderation_enabled: Moderation notifications
    - system_enabled: System notifications
    """
    # Get or create settings
    settings = await crud_notification.notification_settings.get_or_create(
        db, user_id=current_user.id
    )
    
    # Update only provided fields
    update_data = settings_update.model_dump(exclude_unset=True)
    
    if update_data:
        settings = await crud_notification.notification_settings.update_settings(
            db, user_id=current_user.id, settings_data=update_data
        )
    
    return NotificationSettingsResponse(
        email_enabled=settings.email_enabled,
        push_enabled=settings.push_enabled,
        in_app_enabled=settings.in_app_enabled,
        likes_enabled=settings.likes_enabled,
        comments_enabled=settings.comments_enabled,
        follows_enabled=settings.follows_enabled,
        mentions_enabled=settings.mentions_enabled,
        live_streams_enabled=settings.live_streams_enabled,
        subscriptions_enabled=settings.subscriptions_enabled,
        payments_enabled=settings.payments_enabled,
        moderation_enabled=settings.moderation_enabled,
        system_enabled=settings.system_enabled,
    )


# ============================================================================
# PUSH TOKEN ENDPOINTS
# ============================================================================


@router.post(
    "/notifications/push-tokens",
    response_model=PushTokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register push token",
    description="Register a push notification token for mobile/web",
)
async def register_push_token(
    token_data: PushTokenCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PushTokenResponse:
    """
    Register push notification token.
    
    Registers a Firebase Cloud Messaging (FCM) or Apple Push Notification
    Service (APNS) token for sending push notifications to user's device.
    
    **Supported Device Types:**
    - ios: iOS devices (APNS)
    - android: Android devices (FCM)
    - web: Web browsers (FCM Web Push)
    
    **Process:**
    - If token exists, updates last_used_at
    - If token is new, creates new registration
    - Automatically associates with current user
    """
    push_token = await crud_notification.push_token.get_or_create(
        db,
        user_id=current_user.id,
        token=token_data.token,
        device_type=token_data.device_type,
        device_name=token_data.device_name,
    )
    
    return PushTokenResponse(
        id=str(push_token.id),
        token=push_token.token,
        device_type=push_token.device_type,
        device_name=push_token.device_name,
        created_at=push_token.created_at,
        last_used_at=push_token.last_used_at,
    )


@router.get(
    "/notifications/push-tokens",
    response_model=list[PushTokenResponse],
    summary="List push tokens",
    description="Get user's registered push tokens",
)
async def list_push_tokens(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[PushTokenResponse]:
    """
    List registered push tokens.
    
    Returns all push notification tokens registered for the user.
    Useful for managing multiple devices.
    """
    tokens = await crud_notification.push_token.get_by_user(
        db, user_id=current_user.id
    )
    
    return [
        PushTokenResponse(
            id=str(t.id),
            token=t.token,
            device_type=t.device_type,
            device_name=t.device_name,
            created_at=t.created_at,
            last_used_at=t.last_used_at,
        )
        for t in tokens
    ]


@router.delete(
    "/notifications/push-tokens/{token_id}",
    summary="Delete push token",
    description="Unregister a push notification token",
)
async def delete_push_token(
    token_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Delete push notification token.
    
    Unregisters a push token, stopping push notifications to that device.
    Used when user logs out or uninstalls app.
    """
    token = await crud_notification.push_token.get(db, token_id)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Push token not found",
        )
    
    # Verify ownership
    if token.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this token",
        )
    
    await crud_notification.push_token.delete(db, token_id)
    
    return {
        "success": True,
        "message": "Push token deleted",
    }
