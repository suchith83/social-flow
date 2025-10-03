"""
Enhanced Notification Service

Comprehensive notification system with multi-channel delivery:
- In-app notifications
- Email notifications
- Push notifications
- WebSocket real-time delivery
- User preferences
- Template management
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import logging
import json

from app.notifications.models.notification import Notification, NotificationType
from app.notifications.models.extended import (
    NotificationPreference,
    NotificationTemplate,
    EmailLog,
    PushNotificationToken
)
from app.models.user import User

logger = logging.getLogger(__name__)


class EnhancedNotificationService:
    """
    Enhanced notification service with multi-channel delivery.
    
    Features:
    - Multi-channel delivery (in-app, email, push, SMS)
    - User preferences
    - Template management
    - Real-time WebSocket delivery
    - Delivery tracking
    - Quiet hours support
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        channels: List[str] = None,
        priority: str = "normal",
        action_url: str = None,
        action_label: str = None,
        related_id: str = None,
        related_type: str = None,
        actor_id: str = None,
        icon: str = None,
        image_url: str = None,
        metadata: Dict[str, Any] = None,
        expires_hours: int = None
    ) -> Notification:
        """
        Create and deliver a notification.
        
        Args:
            user_id: Recipient user ID
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            channels: Delivery channels (default: in_app only)
            priority: Priority level (low, normal, high, urgent)
            action_url: Optional action URL
            action_label: Optional action button label
            related_id: Related entity ID
            related_type: Related entity type
            actor_id: User who triggered notification
            icon: Notification icon
            image_url: Optional image URL
            metadata: Additional metadata
            expires_hours: Hours until expiration
        
        Returns:
            Created notification
        """
        
        # Default channels
        if channels is None:
            channels = ["in_app"]
        
        # Calculate expiry
        expires_at = None
        if expires_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        
        # Create notification
        notification = Notification(
            user_id=user_id,
            notification_type=notification_type,
            title=title,
            message=message,
            channels=channels,
            priority=priority,
            action_url=action_url,
            action_text=action_label,
            related_id=related_id,
            entity_type=related_type,
            sender_id=actor_id,
            icon=icon,
            image_url=image_url,
            notification_metadata=json.dumps(metadata) if metadata else None,
            expires_at=expires_at,
            status="unread"
        )
        
        self.db.add(notification)
        await self.db.commit()
        await self.db.refresh(notification)
        
        # Deliver through channels
        await self._deliver_notification(notification, channels)
        
        return notification
    
    async def create_from_template(
        self,
        user_id: str,
        template_type: str,
        variables: Dict[str, Any] = None,
        actor_id: str = None,
        related_id: str = None,
        related_type: str = None
    ) -> Optional[Notification]:
        """
        Create notification from template.
        
        Args:
            user_id: Recipient user ID
            template_type: Template type
            variables: Template variables for substitution
            actor_id: User who triggered notification
            related_id: Related entity ID
            related_type: Related entity type
        
        Returns:
            Created notification or None if template not found
        """
        
        # Get template
        result = await self.db.execute(
            select(NotificationTemplate).where(
                and_(
                    NotificationTemplate.type == template_type,
                    NotificationTemplate.is_active == True
                )
            )
        )
        template = result.scalar_one_or_none()
        
        if not template:
            logger.warning(f"Template not found: {template_type}")
            return None
        
        # Substitute variables
        variables = variables or {}
        title = template.title_template.format(**variables)
        message = template.message_template.format(**variables)
        
        # Calculate expiry
        expires_hours = template.default_expires_hours
        
        # Create notification
        return await self.create_notification(
            user_id=user_id,
            notification_type=template_type,
            title=title,
            message=message,
            channels=template.default_channels,
            priority=template.default_priority,
            action_label=template.default_action_label,
            related_id=related_id,
            related_type=related_type,
            actor_id=actor_id,
            icon=template.default_icon,
            expires_hours=expires_hours
        )
    
    async def _deliver_notification(
        self,
        notification: Notification,
        channels: List[str]
    ) -> None:
        """
        Deliver notification through specified channels.
        
        Args:
            notification: Notification to deliver
            channels: Delivery channels
        """
        
        # Check user preferences
        preferences = await self._get_user_preferences(notification.user_id)
        
        # Check quiet hours
        if preferences and preferences.quiet_hours_enabled:
            if self._is_quiet_hours(preferences):
                logger.info(f"Skipping delivery during quiet hours: {notification.id}")
                return
        
        # Deliver through each channel
        for channel in channels:
            if channel == "in_app":
                # In-app already created
                # Send via WebSocket if user is connected
                await self._send_websocket(notification)
            
            elif channel == "email":
                if preferences and preferences.email_enabled:
                    await self._send_email(notification)
            
            elif channel == "push":
                if preferences and preferences.push_enabled:
                    await self._send_push(notification)
            
            elif channel == "sms":
                if preferences and preferences.sms_enabled:
                    await self._send_sms(notification)
    
    async def _get_user_preferences(
        self,
        user_id: str
    ) -> Optional[NotificationPreference]:
        """Get user notification preferences"""
        
        result = await self.db.execute(
            select(NotificationPreference).where(
                NotificationPreference.user_id == user_id
            )
        )
        return result.scalar_one_or_none()
    
    def _is_quiet_hours(self, preferences: NotificationPreference) -> bool:
        """Check if current time is in quiet hours"""
        
        if not preferences.quiet_hours_start or not preferences.quiet_hours_end:
            return False
        
        now = datetime.now(timezone.utc)
        current_time = now.strftime("%H:%M")
        
        start = preferences.quiet_hours_start
        end = preferences.quiet_hours_end
        
        # Handle overnight quiet hours (e.g., 22:00 to 06:00)
        if start > end:
            return current_time >= start or current_time <= end
        else:
            return start <= current_time <= end
    
    async def _send_websocket(self, notification: Notification) -> None:
        """
        Send notification via WebSocket.
        
        Broadcasts to user's active WebSocket connections.
        """
        
        # TODO: Implement WebSocket broadcast
        # This will be implemented in websocket_handler.py
        logger.info(f"WebSocket delivery queued: {notification.id}")
    
    async def _send_email(self, notification: Notification) -> None:
        """
        Send notification via email.
        
        Creates email log and sends via email service.
        """
        
        try:
            # Get user email
            result = await self.db.execute(
                select(User).where(User.id == notification.user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user or not user.email:
                logger.warning(f"No email for user: {notification.user_id}")
                return
            
            # Create email log
            email_log = EmailLog(
                notification_id=notification.id,
                user_id=notification.user_id,
                to_email=user.email,
                subject=notification.title,
                body_text=notification.message,
                body_html=self._render_email_html(notification),
                provider="sendgrid",
                status="pending"
            )
            
            self.db.add(email_log)
            await self.db.commit()
            
            # TODO: Send via email service (SendGrid/SES)
            logger.info(f"Email queued: {email_log.id}")
            
            # Update notification
            notification.is_email_sent = True
            notification.email_sent_at = datetime.now(timezone.utc)
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Email delivery failed: {e}")
    
    async def _send_push(self, notification: Notification) -> None:
        """
        Send notification via push notification.
        
        Sends to all user's active devices.
        """
        
        try:
            # Get user's push tokens
            result = await self.db.execute(
                select(PushNotificationToken).where(
                    and_(
                        PushNotificationToken.user_id == notification.user_id,
                        PushNotificationToken.is_active == True
                    )
                )
            )
            tokens = result.scalars().all()
            
            if not tokens:
                logger.info(f"No push tokens for user: {notification.user_id}")
                return
            
            # TODO: Send via FCM/APNS
            for token in tokens:
                logger.info(f"Push notification queued: {token.device_type}")
            
            # Update notification
            notification.is_push_sent = True
            notification.push_sent_at = datetime.now(timezone.utc)
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Push delivery failed: {e}")
    
    async def _send_sms(self, notification: Notification) -> None:
        """Send notification via SMS (Twilio)"""
        
        # TODO: Implement SMS delivery
        logger.info(f"SMS delivery queued: {notification.id}")
    
    def _render_email_html(self, notification: Notification) -> str:
        """Render email HTML template"""
        
        # TODO: Use proper email template engine
        return f"""
        <html>
            <body>
                <h2>{notification.title}</h2>
                <p>{notification.message}</p>
                {f'<a href="{notification.action_url}">{notification.action_text}</a>' if notification.action_url else ''}
            </body>
        </html>
        """
    
    async def mark_as_read(
        self,
        notification_id: str,
        user_id: str
    ) -> Optional[Notification]:
        """Mark notification as read"""
        
        result = await self.db.execute(
            select(Notification).where(
                and_(
                    Notification.id == notification_id,
                    Notification.user_id == user_id
                )
            )
        )
        notification = result.scalar_one_or_none()
        
        if notification and notification.status == "unread":
            notification.status = "read"
            notification.read_at = datetime.now(timezone.utc)
            await self.db.commit()
        
        return notification
    
    async def mark_all_as_read(self, user_id: str) -> int:
        """Mark all user notifications as read"""
        
        result = await self.db.execute(
            select(Notification).where(
                and_(
                    Notification.user_id == user_id,
                    Notification.status == "unread"
                )
            )
        )
        notifications = result.scalars().all()
        
        count = 0
        now = datetime.now(timezone.utc)
        for notification in notifications:
            notification.status = "read"
            notification.read_at = now
            count += 1
        
        await self.db.commit()
        return count
    
    async def delete_notification(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Delete notification"""
        
        result = await self.db.execute(
            select(Notification).where(
                and_(
                    Notification.id == notification_id,
                    Notification.user_id == user_id
                )
            )
        )
        notification = result.scalar_one_or_none()
        
        if notification:
            await self.db.delete(notification)
            await self.db.commit()
            return True
        
        return False
    
    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        unread_only: bool = False
    ) -> List[Notification]:
        """Get user notifications"""
        
        query = select(Notification).where(Notification.user_id == user_id)
        
        if unread_only:
            query = query.where(Notification.status == "unread")
        
        query = query.order_by(Notification.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications"""
        
        result = await self.db.execute(
            select(func.count(Notification.id)).where(
                and_(
                    Notification.user_id == user_id,
                    Notification.status == "unread"
                )
            )
        )
        return result.scalar()
    
    async def create_or_update_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> NotificationPreference:
        """Create or update user notification preferences"""
        
        # Get existing preferences
        result = await self.db.execute(
            select(NotificationPreference).where(
                NotificationPreference.user_id == user_id
            )
        )
        pref = result.scalar_one_or_none()
        
        if pref:
            # Update existing
            for key, value in preferences.items():
                if hasattr(pref, key):
                    setattr(pref, key, value)
        else:
            # Create new
            pref = NotificationPreference(user_id=user_id, **preferences)
            self.db.add(pref)
        
        await self.db.commit()
        await self.db.refresh(pref)
        return pref
    
    async def get_preferences(
        self,
        user_id: str
    ) -> Optional[NotificationPreference]:
        """Get user notification preferences"""
        
        return await self._get_user_preferences(user_id)
    
    async def register_push_token(
        self,
        user_id: str,
        token: str,
        device_id: str,
        device_type: str,
        device_name: str = None
    ) -> PushNotificationToken:
        """Register push notification token"""
        
        # Check if token already exists
        result = await self.db.execute(
            select(PushNotificationToken).where(
                PushNotificationToken.token == token
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing
            existing.user_id = user_id
            existing.device_id = device_id
            existing.device_type = device_type
            existing.device_name = device_name
            existing.is_active = True
            existing.last_used_at = datetime.now(timezone.utc)
            await self.db.commit()
            return existing
        
        # Create new
        push_token = PushNotificationToken(
            user_id=user_id,
            token=token,
            device_id=device_id,
            device_type=device_type,
            device_name=device_name,
            is_active=True,
            last_used_at=datetime.now(timezone.utc)
        )
        
        self.db.add(push_token)
        await self.db.commit()
        await self.db.refresh(push_token)
        return push_token
    
    async def unregister_push_token(
        self,
        token: str
    ) -> bool:
        """Unregister push notification token"""
        
        result = await self.db.execute(
            select(PushNotificationToken).where(
                PushNotificationToken.token == token
            )
        )
        push_token = result.scalar_one_or_none()
        
        if push_token:
            push_token.is_active = False
            await self.db.commit()
            return True
        
        return False

