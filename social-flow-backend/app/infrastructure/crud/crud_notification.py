"""
CRUD operations for notification models (Notification, NotificationSettings, PushToken).
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud.base import CRUDBase
from app.models.notification import (
    Notification,
    NotificationSettings,
    PushToken,
    NotificationType,
)
from app.schemas.base import BaseSchema


class CRUDNotification(CRUDBase[Notification, BaseSchema, BaseSchema]):
    """CRUD operations for Notification model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
        unread_only: bool = False,
        notification_type: Optional[NotificationType] = None,
    ) -> List[Notification]:
        """Get notifications for a user."""
        query = select(self.model).where(self.model.user_id == user_id)
        
        if unread_only:
            query = query.where(self.model.is_read == False)
        
        if notification_type:
            query = query.where(self.model.notification_type == notification_type)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def mark_as_read(
        self,
        db: AsyncSession,
        *,
        notification_id: UUID,
    ) -> Optional[Notification]:
        """Mark a notification as read."""
        notification = await self.get(db, notification_id)
        if not notification:
            return None
        
        notification.is_read = True
        notification.read_at = datetime.now(timezone.utc)
        db.add(notification)
        await db.commit()
        await db.refresh(notification)
        return notification

    async def mark_all_as_read(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> int:
        """Mark all notifications as read for a user."""
        from sqlalchemy import update as sql_update
        
        query = (
            sql_update(self.model)
            .where(
                and_(
                    self.model.user_id == user_id,
                    self.model.is_read == False,
                )
            )
            .values(
                is_read=True,
                read_at=datetime.now(timezone.utc),
            )
        )
        result = await db.execute(query)
        await db.commit()
        return result.rowcount

    async def get_unread_count(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> int:
        """Get count of unread notifications for a user."""
        query = (
            select(func.count())
            .select_from(self.model)
            .where(
                and_(
                    self.model.user_id == user_id,
                    self.model.is_read == False,
                )
            )
        )
        result = await db.execute(query)
        return result.scalar_one()

    async def delete_old_notifications(
        self,
        db: AsyncSession,
        *,
        days: int = 30,
    ) -> int:
        """Delete notifications older than specified days."""
        from datetime import timedelta
        from sqlalchemy import delete as sql_delete
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = sql_delete(self.model).where(
            and_(
                self.model.created_at < cutoff_date,
                self.model.is_read == True,
            )
        )
        result = await db.execute(query)
        await db.commit()
        return result.rowcount

    async def create_bulk(
        self,
        db: AsyncSession,
        *,
        user_ids: List[UUID],
        notification_type: NotificationType,
        title: str,
        message: str,
        data: Optional[dict] = None,
    ) -> List[Notification]:
        """Create notifications for multiple users."""
        notifications = []
        for user_id in user_ids:
            notification = Notification(
                user_id=user_id,
                notification_type=notification_type,
                title=title,
                message=message,
                data=data,
            )
            db.add(notification)
            notifications.append(notification)
        
        await db.commit()
        for notification in notifications:
            await db.refresh(notification)
        
        return notifications


class CRUDNotificationSettings(CRUDBase[NotificationSettings, BaseSchema, BaseSchema]):
    """CRUD operations for NotificationSettings model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> Optional[NotificationSettings]:
        """Get notification settings for a user."""
        return await self.get_by_field(db, "user_id", user_id)

    async def get_or_create(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> NotificationSettings:
        """Get existing settings or create default ones."""
        settings = await self.get_by_user(db, user_id=user_id)
        
        if settings:
            return settings
        
        # Create default settings
        settings = NotificationSettings(
            user_id=user_id,
            email_enabled=True,
            push_enabled=True,
            in_app_enabled=True,
            likes_enabled=True,
            comments_enabled=True,
            follows_enabled=True,
            mentions_enabled=True,
            live_streams_enabled=True,
            subscriptions_enabled=True,
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
        return settings

    async def update_settings(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        settings_data: dict,
    ) -> Optional[NotificationSettings]:
        """Update notification settings for a user."""
        settings = await self.get_by_user(db, user_id=user_id)
        if not settings:
            return None
        
        for field, value in settings_data.items():
            if hasattr(settings, field):
                setattr(settings, field, value)
        
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
        return settings

    async def is_notification_enabled(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        notification_type: NotificationType,
        channel: str = "in_app",  # 'email', 'push', 'in_app'
    ) -> bool:
        """Check if a specific notification type is enabled for a user."""
        settings = await self.get_by_user(db, user_id=user_id)
        if not settings:
            return True  # Default to enabled if no settings
        
        # Check channel
        channel_enabled = {
            "email": settings.email_enabled,
            "push": settings.push_enabled,
            "in_app": settings.in_app_enabled,
        }.get(channel, True)
        
        if not channel_enabled:
            return False
        
        # Check notification type
        type_mapping = {
            NotificationType.LIKE: settings.likes_enabled,
            NotificationType.COMMENT: settings.comments_enabled,
            NotificationType.FOLLOW: settings.follows_enabled,
            NotificationType.MENTION: settings.mentions_enabled,
            NotificationType.LIVE_STREAM: settings.live_streams_enabled,
            NotificationType.SUBSCRIPTION: settings.subscriptions_enabled,
        }
        
        return type_mapping.get(notification_type, True)


class CRUDPushToken(CRUDBase[PushToken, BaseSchema, BaseSchema]):
    """CRUD operations for PushToken model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PushToken]:
        """Get push tokens for a user."""
        query = (
            select(self.model)
            .where(self.model.user_id == user_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_by_token(
        self,
        db: AsyncSession,
        *,
        token: str,
    ) -> Optional[PushToken]:
        """Get push token by token string."""
        return await self.get_by_field(db, "token", token)

    async def get_or_create(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        token: str,
        device_type: str,
        device_name: Optional[str] = None,
    ) -> PushToken:
        """Get existing token or create new one."""
        existing = await self.get_by_token(db, token=token)
        
        if existing:
            # Update last used timestamp
            existing.last_used_at = datetime.now(timezone.utc)
            db.add(existing)
            await db.commit()
            await db.refresh(existing)
            return existing
        
        # Create new token
        push_token = PushToken(
            user_id=user_id,
            token=token,
            device_type=device_type,
            device_name=device_name,
        )
        db.add(push_token)
        await db.commit()
        await db.refresh(push_token)
        return push_token

    async def update_last_used(
        self,
        db: AsyncSession,
        *,
        token_id: UUID,
    ) -> Optional[PushToken]:
        """Update last used timestamp for a token."""
        token = await self.get(db, token_id)
        if not token:
            return None
        
        token.last_used_at = datetime.now(timezone.utc)
        db.add(token)
        await db.commit()
        await db.refresh(token)
        return token

    async def delete_by_token(
        self,
        db: AsyncSession,
        *,
        token: str,
    ) -> bool:
        """Delete a push token by token string."""
        from sqlalchemy import delete as sql_delete
        
        query = sql_delete(self.model).where(self.model.token == token)
        result = await db.execute(query)
        await db.commit()
        return result.rowcount > 0

    async def delete_inactive_tokens(
        self,
        db: AsyncSession,
        *,
        days: int = 90,
    ) -> int:
        """Delete tokens not used in specified days."""
        from datetime import timedelta
        from sqlalchemy import delete as sql_delete
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = sql_delete(self.model).where(
            or_(
                self.model.last_used_at < cutoff_date,
                and_(
                    self.model.last_used_at.is_(None),
                    self.model.created_at < cutoff_date,
                )
            )
        )
        result = await db.execute(query)
        await db.commit()
        return result.rowcount


# Create singleton instances
notification = CRUDNotification(Notification)
notification_settings = CRUDNotificationSettings(NotificationSettings)
push_token = CRUDPushToken(PushToken)
