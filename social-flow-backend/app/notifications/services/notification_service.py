"""
Notification Service for handling all notification operations.

This service integrates all existing notification modules from
the mobile-backend and other services into the FastAPI application.
"""

import logging
from typing import Any, Dict, List
from datetime import datetime
import uuid
import json

from app.core.exceptions import NotificationServiceError
from app.core.redis import get_cache

logger = logging.getLogger(__name__)


class NotificationService:
    """Main notification service integrating all notification capabilities."""

    def __init__(self):
        self.cache = None
        self._initialize_services()

    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            self.cache = await get_cache()
        return self.cache

    def _initialize_services(self):
        """Initialize notification services."""
        try:
            # Initialize Firebase Cloud Messaging (FCM)
            try:
                import firebase_admin
                from firebase_admin import credentials, messaging
                
                # Initialize Firebase if credentials exist
                if not firebase_admin._apps:
                    # Try to initialize from environment or JSON file
                    try:
                        cred = credentials.ApplicationDefault()
                        firebase_admin.initialize_app(cred)
                        self.fcm = messaging
                        logger.info("Firebase Cloud Messaging initialized")
                    except Exception as e:
                        logger.warning(f"FCM initialization skipped: {e}")
                        self.fcm = None
                else:
                    self.fcm = messaging
            except ImportError:
                logger.warning("firebase-admin not installed")
                self.fcm = None
            
            # Initialize email service
            try:
                from app.services.email_service import email_service
                self.email_service = email_service
                logger.info("Email service connected")
            except Exception as e:
                logger.warning(f"Email service connection failed: {e}")
                self.email_service = None
            
            # Initialize SMS service (Twilio)
            try:
                from twilio.rest import Client
                from app.core.config import settings
                
                if hasattr(settings, 'TWILIO_ACCOUNT_SID') and settings.TWILIO_ACCOUNT_SID:
                    self.sms_client = Client(
                        settings.TWILIO_ACCOUNT_SID,
                        settings.TWILIO_AUTH_TOKEN
                    )
                    self.twilio_phone = settings.TWILIO_PHONE_NUMBER
                    logger.info("Twilio SMS service initialized")
                else:
                    self.sms_client = None
            except ImportError:
                logger.warning("twilio not installed")
                self.sms_client = None
            except Exception as e:
                logger.warning(f"SMS service initialization failed: {e}")
                self.sms_client = None
            
            logger.info("Notification Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Notification Service: {e}")
            raise NotificationServiceError(f"Failed to initialize Notification Service: {e}")

    async def send_push_notification(self, user_id: str, title: str, body: str, 
                                   data: Dict[str, Any] = None, 
                                   notification_type: str = "general") -> Dict[str, Any]:
        """Send a push notification to a user."""
        try:
            notification_id = str(uuid.uuid4())
            
            if self.fcm:
                # Get user's device tokens (from database)
                # For now, we'll assume device_token is passed in data
                device_token = data.get("device_token") if data else None
                
                if device_token:
                    # Create FCM message
                    message = self.fcm.Message(
                        notification=self.fcm.Notification(
                            title=title,
                            body=body
                        ),
                        data=data or {},
                        token=device_token
                    )
                    
                    # Send message
                    try:
                        response = self.fcm.send(message)
                        status = "sent"
                        logger.info(f"Push notification sent: {response}")
                    except Exception as e:
                        status = "failed"
                        logger.error(f"Failed to send push notification: {e}")
                else:
                    status = "no_device_token"
            else:
                status = "fcm_not_initialized"
            
            # Store notification in database
            notification_data = {
                "notification_id": notification_id,
                "user_id": user_id,
                "title": title,
                "body": body,
                "data": data or {},
                "type": notification_type,
                "status": status,
                "sent_at": datetime.utcnow().isoformat()
            }
            
            # Save to database (cache for now)
            cache = await self._get_cache()
            if cache:
                await cache.setex(
                    f"notification:{notification_id}",
                    86400,  # 24 hours
                    json.dumps(notification_data)
                )
            
            return notification_data
        except Exception as e:
            raise NotificationServiceError(f"Failed to send push notification: {str(e)}")

    async def send_email_notification(self, user_id: str, subject: str, body: str, 
                                    template: str = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send an email notification to a user."""
        try:
            notification_id = str(uuid.uuid4())
            
            if self.email_service:
                # Get user's email address (from data or database)
                to_email = data.get("email") if data else None
                
                if to_email:
                    try:
                        # Send email using email service
                        await self.email_service.send_email(
                            to_email=to_email,
                            subject=subject,
                            html_content=body,
                            text_content=body
                        )
                        status = "sent"
                        logger.info(f"Email notification sent to {to_email}")
                    except Exception as e:
                        status = "failed"
                        logger.error(f"Failed to send email: {e}")
                else:
                    status = "no_email_address"
            else:
                status = "email_service_not_initialized"
            
            notification_data = {
                "notification_id": notification_id,
                "user_id": user_id,
                "subject": subject,
                "body": body,
                "template": template,
                "data": data or {},
                "type": "email",
                "status": status,
                "sent_at": datetime.utcnow().isoformat()
            }
            
            # Save to cache
            cache = await self._get_cache()
            if cache:
                await cache.setex(
                    f"notification:{notification_id}",
                    86400,  # 24 hours
                    json.dumps(notification_data)
                )
            
            return notification_data
        except Exception as e:
            raise NotificationServiceError(f"Failed to send email notification: {str(e)}")

    async def send_sms_notification(self, user_id: str, message: str, 
                                  data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send an SMS notification to a user."""
        try:
            notification_id = str(uuid.uuid4())
            
            if self.sms_client:
                # Get user's phone number (from data or database)
                to_phone = data.get("phone") if data else None
                
                if to_phone:
                    try:
                        # Send SMS via Twilio
                        sms_message = self.sms_client.messages.create(
                            body=message,
                            from_=self.twilio_phone,
                            to=to_phone
                        )
                        status = "sent"
                        logger.info(f"SMS sent to {to_phone}: {sms_message.sid}")
                    except Exception as e:
                        status = "failed"
                        logger.error(f"Failed to send SMS: {e}")
                else:
                    status = "no_phone_number"
            else:
                status = "sms_service_not_initialized"
            
            notification_data = {
                "notification_id": notification_id,
                "user_id": user_id,
                "message": message,
                "data": data or {},
                "type": "sms",
                "status": status,
                "sent_at": datetime.utcnow().isoformat()
            }
            
            # Save to cache
            cache = await self._get_cache()
            if cache:
                await cache.setex(
                    f"notification:{notification_id}",
                    86400,  # 24 hours
                    json.dumps(notification_data)
                )
            
            return notification_data
        except Exception as e:
            raise NotificationServiceError(f"Failed to send SMS notification: {str(e)}")

    async def get_user_notifications(self, user_id: str, limit: int = 50, 
                                   offset: int = 0, notification_type: str = None) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        try:
            # TODO: Retrieve notifications from database
            notifications = []
            
            # Placeholder data
            for i in range(min(limit, 10)):
                notifications.append({
                    "notification_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "title": f"Notification {i+1}",
                    "body": f"This is notification {i+1}",
                    "type": notification_type or "general",
                    "status": "sent",
                    "created_at": datetime.utcnow().isoformat()
                })
            
            return notifications
        except Exception as e:
            raise NotificationServiceError(f"Failed to get user notifications: {str(e)}")

    async def mark_notification_read(self, notification_id: str, user_id: str) -> Dict[str, Any]:
        """Mark a notification as read."""
        try:
            # TODO: Update notification status in database
            return {
                "notification_id": notification_id,
                "user_id": user_id,
                "status": "read",
                "read_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise NotificationServiceError(f"Failed to mark notification as read: {str(e)}")

    async def mark_all_notifications_read(self, user_id: str) -> Dict[str, Any]:
        """Mark all notifications as read for a user."""
        try:
            # TODO: Update all notifications for user in database
            return {
                "user_id": user_id,
                "status": "all_read",
                "read_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise NotificationServiceError(f"Failed to mark all notifications as read: {str(e)}")

    async def delete_notification(self, notification_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a notification."""
        try:
            # TODO: Delete notification from database
            return {
                "notification_id": notification_id,
                "user_id": user_id,
                "status": "deleted",
                "deleted_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise NotificationServiceError(f"Failed to delete notification: {str(e)}")

    async def get_notification_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get notification preferences for a user."""
        try:
            # TODO: Retrieve preferences from database
            return {
                "user_id": user_id,
                "push_enabled": True,
                "email_enabled": True,
                "sms_enabled": False,
                "preferences": {
                    "likes": True,
                    "comments": True,
                    "follows": True,
                    "mentions": True,
                    "system": True
                }
            }
        except Exception as e:
            raise NotificationServiceError(f"Failed to get notification preferences: {str(e)}")

    async def update_notification_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update notification preferences for a user."""
        try:
            # TODO: Update preferences in database
            return {
                "user_id": user_id,
                "preferences": preferences,
                "updated_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise NotificationServiceError(f"Failed to update notification preferences: {str(e)}")

    async def send_bulk_notification(self, user_ids: List[str], title: str, body: str, 
                                   notification_type: str = "general", 
                                   data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a notification to multiple users."""
        try:
            notification_id = str(uuid.uuid4())
            
            # TODO: Send bulk notifications
            # This would typically involve:
            # 1. Batching users by notification type
            # 2. Sending notifications in parallel
            # 3. Tracking delivery status
            
            results = []
            for user_id in user_ids:
                result = await self.send_push_notification(
                    user_id, title, body, data, notification_type
                )
                results.append(result)
            
            return {
                "notification_id": notification_id,
                "user_count": len(user_ids),
                "results": results,
                "sent_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise NotificationServiceError(f"Failed to send bulk notification: {str(e)}")

    async def get_notification_stats(self, user_id: str, time_range: str = "30d") -> Dict[str, Any]:
        """Get notification statistics for a user."""
        try:
            # TODO: Calculate stats from database
            return {
                "user_id": user_id,
                "time_range": time_range,
                "total_sent": 0,
                "total_read": 0,
                "total_unread": 0,
                "read_rate": 0.0,
                "by_type": {
                    "push": 0,
                    "email": 0,
                    "sms": 0
                }
            }
        except Exception as e:
            raise NotificationServiceError(f"Failed to get notification stats: {str(e)}")


notification_service = NotificationService()
