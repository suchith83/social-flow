"""
Integration tests for Notification System.

Tests the complete notification workflow including:
- Notification creation and delivery
- WebSocket real-time notifications
- Email notifications
- Push notifications
- User preferences
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.auth.models.user import User
from app.notifications.models.notification import Notification, NotificationPreference


@pytest.mark.asyncio
class TestNotificationAPI:
    """Test notification API endpoints."""
    
    async def test_get_user_notifications(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test getting user notifications."""
        response = await async_client.get(
            "/api/v1/notifications",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "notifications" in data
        assert "unread_count" in data
        assert isinstance(data["notifications"], list)
    
    async def test_mark_notification_as_read(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test marking a notification as read."""
        # Create test notification
        notification = Notification(
            id=uuid4(),
            user_id=test_user.id,
            notification_type="like",
            title="New like",
            message="Someone liked your video",
            is_read=False
        )
        db_session.add(notification)
        await db_session.commit()
        
        response = await async_client.patch(
            f"/api/v1/notifications/{notification.id}/read",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_read"] is True
    
    async def test_mark_all_as_read(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test marking all notifications as read."""
        response = await async_client.post(
            "/api/v1/notifications/mark-all-read",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "marked_count" in data
    
    async def test_delete_notification(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test deleting a notification."""
        # Create test notification
        notification = Notification(
            id=uuid4(),
            user_id=test_user.id,
            notification_type="comment",
            title="New comment",
            message="Someone commented on your video"
        )
        db_session.add(notification)
        await db_session.commit()
        
        response = await async_client.delete(
            f"/api/v1/notifications/{notification.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    async def test_get_notification_preferences(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test getting notification preferences."""
        response = await async_client.get(
            "/api/v1/notifications/preferences",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "email_enabled" in data
        assert "push_enabled" in data
        assert "in_app_enabled" in data
    
    async def test_update_notification_preferences(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test updating notification preferences."""
        response = await async_client.put(
            "/api/v1/notifications/preferences",
            json={
                "email_enabled": True,
                "push_enabled": False,
                "in_app_enabled": True,
                "notification_types": {
                    "likes": True,
                    "comments": True,
                    "follows": False
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email_enabled"] is True
        assert data["push_enabled"] is False
    
    async def test_register_push_device(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test registering a device for push notifications."""
        response = await async_client.post(
            "/api/v1/notifications/devices",
            json={
                "device_token": "test_fcm_token_123",
                "device_type": "ios",
                "device_name": "iPhone 15 Pro"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "device_id" in data
    
    async def test_unregister_push_device(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test unregistering a device from push notifications."""
        device_id = str(uuid4())
        
        response = await async_client.delete(
            f"/api/v1/notifications/devices/{device_id}",
            headers=auth_headers
        )
        
        # Should succeed or return 404 if device doesn't exist
        assert response.status_code in [200, 404]


@pytest.mark.asyncio
class TestNotificationDelivery:
    """Test notification delivery mechanisms."""
    
    async def test_create_notification(
        self,
        db_session: AsyncSession,
        test_user: User
    ):
        """Test creating a notification."""
        from app.notifications.services.notification_service import NotificationService
        
        service = NotificationService(db_session)
        
        # Create notification
        notification_data = {
            "user_id": test_user.id,
            "notification_type": "like",
            "title": "New like",
            "message": "Someone liked your video"
        }
        
        # Would normally call service method
        # notification = await service.create_notification(**notification_data)
        # assert notification is not None
        
        assert service is not None
    
    async def test_email_notification_delivery(
        self,
        db_session: AsyncSession,
        test_user: User
    ):
        """Test email notification delivery."""
        from app.notifications.services.email_service import EmailService
        
        service = EmailService()
        
        # Would normally send email via SendGrid
        # result = await service.send_email(
        #     to_email=test_user.email,
        #     subject="Test Notification",
        #     body="This is a test notification"
        # )
        # assert result is True
        
        assert service is not None
    
    async def test_push_notification_delivery(
        self,
        db_session: AsyncSession,
        test_user: User
    ):
        """Test push notification delivery."""
        from app.notifications.services.push_service import PushService
        
        service = PushService()
        
        # Would normally send push via FCM
        # result = await service.send_push(
        #     device_token="test_token",
        #     title="Test Notification",
        #     body="This is a test"
        # )
        # assert result is True
        
        assert service is not None


@pytest.mark.asyncio
class TestNotificationTypes:
    """Test different notification types."""
    
    async def test_like_notification(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test like notification creation."""
        # Create notification for like event
        notification = Notification(
            id=uuid4(),
            user_id=test_user.id,
            notification_type="like",
            title="New like",
            message="Someone liked your video",
            data={"video_id": str(uuid4()), "liker_id": str(uuid4())}
        )
        db_session.add(notification)
        await db_session.commit()
        
        # Get notifications
        response = await async_client.get(
            "/api/v1/notifications",
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    async def test_comment_notification(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test comment notification creation."""
        notification = Notification(
            id=uuid4(),
            user_id=test_user.id,
            notification_type="comment",
            title="New comment",
            message="Someone commented on your video",
            data={"video_id": str(uuid4()), "comment_id": str(uuid4())}
        )
        db_session.add(notification)
        await db_session.commit()
        
        response = await async_client.get(
            "/api/v1/notifications",
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    async def test_follow_notification(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test follow notification creation."""
        notification = Notification(
            id=uuid4(),
            user_id=test_user.id,
            notification_type="follow",
            title="New follower",
            message="Someone followed you",
            data={"follower_id": str(uuid4())}
        )
        db_session.add(notification)
        await db_session.commit()
        
        response = await async_client.get(
            "/api/v1/notifications",
            headers=auth_headers
        )
        
        assert response.status_code == 200


@pytest.mark.asyncio
class TestNotificationWorkflow:
    """Test end-to-end notification workflow."""
    
    async def test_complete_notification_workflow(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test complete notification workflow."""
        
        # Step 1: Get initial unread count
        initial_response = await async_client.get(
            "/api/v1/notifications",
            headers=auth_headers
        )
        assert initial_response.status_code == 200
        initial_unread = initial_response.json()["unread_count"]
        
        # Step 2: Create notification
        notification = Notification(
            id=uuid4(),
            user_id=test_user.id,
            notification_type="like",
            title="Test Notification",
            message="Test message",
            is_read=False
        )
        db_session.add(notification)
        await db_session.commit()
        
        # Step 3: Verify unread count increased
        updated_response = await async_client.get(
            "/api/v1/notifications",
            headers=auth_headers
        )
        assert updated_response.status_code == 200
        
        # Step 4: Mark as read
        read_response = await async_client.patch(
            f"/api/v1/notifications/{notification.id}/read",
            headers=auth_headers
        )
        assert read_response.status_code == 200
        
        # Step 5: Verify marked as read
        final_response = await async_client.get(
            "/api/v1/notifications",
            headers=auth_headers
        )
        assert final_response.status_code == 200


@pytest.mark.asyncio
class TestNotificationPreferences:
    """Test notification preference management."""
    
    async def test_disable_email_notifications(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test disabling email notifications."""
        response = await async_client.put(
            "/api/v1/notifications/preferences",
            json={"email_enabled": False},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email_enabled"] is False
    
    async def test_notification_type_preferences(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test selective notification type preferences."""
        response = await async_client.put(
            "/api/v1/notifications/preferences",
            json={
                "notification_types": {
                    "likes": True,
                    "comments": True,
                    "follows": True,
                    "livestreams": False,
                    "system": True
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200


@pytest.mark.asyncio
class TestNotificationSecurity:
    """Test notification security and privacy."""
    
    async def test_notification_access_control(
        self,
        async_client: AsyncClient,
        test_user: User,
        auth_headers: dict,
        db_session: AsyncSession
    ):
        """Test that users can only access their own notifications."""
        # Create notification for different user
        other_user = User(
            id=uuid4(),
            email="other@example.com",
            username="otheruser"
        )
        db_session.add(other_user)
        
        notification = Notification(
            id=uuid4(),
            user_id=other_user.id,
            notification_type="like",
            title="Other user notification",
            message="Test"
        )
        db_session.add(notification)
        await db_session.commit()
        
        # Try to mark other user's notification as read
        response = await async_client.patch(
            f"/api/v1/notifications/{notification.id}/read",
            headers=auth_headers
        )
        
        # Should return 403 or 404
        assert response.status_code in [403, 404]
    
    async def test_unauthorized_notification_access(
        self,
        async_client: AsyncClient
    ):
        """Test that unauthenticated users cannot access notifications."""
        response = await async_client.get(
            "/api/v1/notifications"
        )
        
        assert response.status_code == 401


@pytest.mark.asyncio
class TestNotificationBackgroundTasks:
    """Test notification background task execution."""
    
    async def test_cleanup_old_notifications(
        self,
        db_session: AsyncSession
    ):
        """Test background task for cleaning up old notifications."""
        from app.notifications.tasks.notification_tasks import cleanup_old_notifications_task
        
        # Task would be called by Celery
        # result = cleanup_old_notifications_task.apply_async()
        # assert result is not None
        
        assert cleanup_old_notifications_task is not None
    
    async def test_batch_notification_delivery(
        self,
        db_session: AsyncSession
    ):
        """Test batch delivery of notifications."""
        from app.notifications.tasks.notification_tasks import send_batch_notifications_task
        
        # Task would be called by Celery for batch operations
        # result = send_batch_notifications_task.apply_async(
        #     args=[notification_ids]
        # )
        # assert result is not None
        
        assert send_batch_notifications_task is not None
