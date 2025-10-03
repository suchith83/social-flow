"""
Comprehensive integration tests for video management endpoints.

Tests cover:
- Video upload workflow (initiate, complete)
- Video listing and discovery (list, trending, search, my videos)
- Video CRUD operations (get, update, delete)
- Video streaming and playback
- Video interactions (views, likes)
- Video analytics
- Admin moderation (approve, reject)
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud import user as crud_user, video as crud_video
from app.models.user import User, UserRole, UserStatus
from app.models.video import VideoStatus
from app.schemas.video import VideoCreate


class TestVideoUpload:
    """Test video upload workflow."""
    
    @pytest.mark.asyncio
    async def test_initiate_video_upload_as_creator(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test initiating video upload as creator."""
        # Arrange - Make user a creator
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            "/api/v1/videos",
            headers=headers,
            json={
                "filename": "test_video.mp4",
                "file_size": 50000000,  # 50MB
                "content_type": "video/mp4",
            },
        )
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert "upload_url" in data
        assert "video_id" in data
        assert "expires_in" in data
    
    @pytest.mark.asyncio
    async def test_initiate_upload_non_creator_fails(
        self,
        async_client: AsyncClient,
        test_user: User,
    ):
        """Test non-creator cannot initiate upload."""
        # Arrange - Login as regular user
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            "/api/v1/videos",
            headers=headers,
            json={
                "filename": "test_video.mp4",
                "file_size": 50000000,
                "content_type": "video/mp4",
            },
        )
        
        # Assert
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_complete_video_upload(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test completing video upload with metadata."""
        # Arrange - Create video as creator
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Temp Title",
            original_filename="test.mp4",
            file_size=50000000,
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            f"/api/v1/videos/{video.id}/complete",
            headers=headers,
            json={
                "title": "My Awesome Video",
                "description": "This is a test video",
                "tags": ["test", "demo"],
                "visibility": "public",
            },
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "My Awesome Video"
        assert data["description"] == "This is a test video"
        assert "test" in data["tags"]


class TestVideoListing:
    """Test video listing and discovery endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_public_videos(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test listing public videos with pagination."""
        # Arrange - Create public videos
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        for i in range(3):
            video_data = VideoCreate(
                title=f"Public Video {i}",
                description="Test description",
                original_filename=f"video{i}.mp4",
                file_size=50000000,
                visibility="public",
            )
            video = await crud_video.create_with_owner(
                db_session,
                obj_in=video_data,
                owner_id=test_user.id,
            )
            # Set video to PROCESSED status so it appears in listings
            video.status = VideoStatus.PROCESSED
            db_session.add(video)
        await db_session.commit()
        
        # Act
        response = await async_client.get("/api/v1/videos?skip=0&limit=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert len(data["items"]) >= 3
    
    @pytest.mark.asyncio
    async def test_get_trending_videos(
        self,
        async_client: AsyncClient,
    ):
        """Test getting trending videos."""
        # Act
        response = await async_client.get("/api/v1/videos/trending?limit=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    @pytest.mark.asyncio
    async def test_search_videos_by_title(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test searching videos by title."""
        # Arrange - Create searchable video
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Python Tutorial",
            description="Learn Python",
            original_filename="python.mp4",
            file_size=50000000,
            visibility="public",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        # Set video to PROCESSED status so it appears in search
        video.status = VideoStatus.PROCESSED
        db_session.add(video)
        await db_session.commit()
        
        # Act
        response = await async_client.get("/api/v1/videos/search?q=Python")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1
    
    @pytest.mark.asyncio
    async def test_get_my_videos(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test getting current user's videos."""
        # Arrange - Create user's video
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="My Video",
            original_filename="my_video.mp4",
            file_size=50000000,
        )
        await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.get("/api/v1/videos/my", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 1


class TestVideoCRUD:
    """Test video CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_get_public_video_by_id(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test getting public video details."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Public Video",
            original_filename="public.mp4",
            file_size=50000000,
            visibility="public",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Act
        response = await async_client.get(f"/api/v1/videos/{video.id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Public Video"
    
    @pytest.mark.asyncio
    async def test_get_private_video_by_owner(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test owner can view their private video."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Private Video",
            original_filename="private.mp4",
            file_size=50000000,
            visibility="private",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.get(f"/api/v1/videos/{video.id}", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Private Video"
    
    @pytest.mark.asyncio
    async def test_get_private_video_by_non_owner_fails(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test non-owner cannot view private video."""
        # Arrange - Create private video
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Private Video",
            original_filename="private.mp4",
            file_size=50000000,
            visibility="private",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Create another user
        from app.schemas.user import UserRegister
        other_user_data = UserRegister(
            username="other_user",
            email="other@example.com",
            password="TestPassword123",
        )
        other_user = await crud_user.create(db_session, obj_in=other_user_data)
        other_user.status = UserStatus.ACTIVE
        other_user.is_verified = True
        db_session.add(other_user)
        await db_session.commit()
        
        # Login as other user
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": other_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.get(f"/api/v1/videos/{video.id}", headers=headers)
        
        # Assert
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_update_video(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test updating video metadata."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Original Title",
            original_filename="video.mp4",
            file_size=50000000,
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.put(
            f"/api/v1/videos/{video.id}",
            headers=headers,
            json={
                "title": "Updated Title",
                "description": "Updated description",
            },
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
    
    @pytest.mark.asyncio
    async def test_delete_video(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test deleting own video."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Video to Delete",
            original_filename="delete.mp4",
            file_size=50000000,
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.delete(f"/api/v1/videos/{video.id}", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestVideoStreaming:
    """Test video streaming functionality."""
    
    @pytest.mark.asyncio
    async def test_get_streaming_urls(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test getting video streaming URLs."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Streaming Video",
            original_filename="stream.mp4",
            file_size=50000000,
            visibility="public",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        # Set video to PROCESSED status so it's ready for streaming
        video.status = VideoStatus.PROCESSED
        db_session.add(video)
        await db_session.commit()
        
        # Act
        response = await async_client.get(f"/api/v1/videos/{video.id}/stream")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "hls_url" in data or "dash_url" in data or "progressive_url" in data


class TestVideoInteractions:
    """Test video interactions (views, likes)."""
    
    @pytest.mark.asyncio
    async def test_record_video_view(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test recording a video view."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Video for Views",
            original_filename="view.mp4",
            file_size=50000000,
            visibility="public",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(f"/api/v1/videos/{video.id}/view", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_like_video(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test liking a video."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Video to Like",
            original_filename="like.mp4",
            file_size=50000000,
            visibility="public",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(f"/api/v1/videos/{video.id}/like", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_unlike_video(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test unliking a video."""
        # Arrange - Create and like video
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Video to Unlike",
            original_filename="unlike.mp4",
            file_size=50000000,
            visibility="public",
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Like first
        await async_client.post(f"/api/v1/videos/{video.id}/like", headers=headers)
        
        # Act - Unlike
        response = await async_client.delete(f"/api/v1/videos/{video.id}/like", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestVideoAnalytics:
    """Test video analytics endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_video_analytics(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test getting video analytics."""
        # Arrange
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Analytics Video",
            original_filename="analytics.mp4",
            file_size=50000000,
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.get(f"/api/v1/videos/{video.id}/analytics", headers=headers)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "views" in data or "total_views" in data


class TestVideoModeration:
    """Test admin moderation endpoints."""
    
    @pytest.mark.asyncio
    async def test_admin_approve_video(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test admin can approve videos."""
        # Arrange - Create admin
        from app.schemas.user import UserRegister
        admin_data = UserRegister(
            username="admin_user",
            email="admin@example.com",
            password="AdminPassword123!",
        )
        admin_user = await crud_user.create(db_session, obj_in=admin_data)
        admin_user.role = UserRole.ADMIN
        admin_user.status = UserStatus.ACTIVE
        admin_user.is_verified = True
        db_session.add(admin_user)
        await db_session.commit()
        
        # Create video
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Video to Approve",
            original_filename="approve.mp4",
            file_size=50000000,
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Admin login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": admin_user.email, "password": "AdminPassword123!"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            f"/api/v1/videos/{video.id}/admin/approve",
            headers=headers,
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_admin_reject_video(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test admin can reject videos."""
        # Arrange - Create admin
        from app.schemas.user import UserRegister
        admin_data = UserRegister(
            username="admin_reject",
            email="adminreject@example.com",
            password="AdminPassword123!",
        )
        admin_user = await crud_user.create(db_session, obj_in=admin_data)
        admin_user.role = UserRole.ADMIN
        admin_user.status = UserStatus.ACTIVE
        admin_user.is_verified = True
        db_session.add(admin_user)
        await db_session.commit()
        
        # Create video
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Video to Reject",
            original_filename="reject.mp4",
            file_size=50000000,
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Admin login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": admin_user.email, "password": "AdminPassword123!"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.post(
            f"/api/v1/videos/{video.id}/admin/reject",
            headers=headers,
            json={"reason": "Inappropriate content"},
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestVideoEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_video(
        self,
        async_client: AsyncClient,
    ):
        """Test getting video that doesn't exist."""
        from uuid import uuid4
        
        # Act
        response = await async_client.get(f"/api/v1/videos/{uuid4()}")
        
        # Assert
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_update_other_users_video_fails(
        self,
        async_client: AsyncClient,
        test_user: User,
        db_session: AsyncSession,
    ):
        """Test cannot update another user's video."""
        # Arrange - Create video owner
        test_user.role = UserRole.CREATOR
        db_session.add(test_user)
        await db_session.commit()
        
        video_data = VideoCreate(
            title="Owner's Video",
            original_filename="owner.mp4",
            file_size=50000000,
        )
        video = await crud_video.create_with_owner(
            db_session,
            obj_in=video_data,
            owner_id=test_user.id,
        )
        
        # Create another user
        from app.schemas.user import UserRegister
        other_user_data = UserRegister(
            username="other_creator",
            email="othercreator@example.com",
            password="TestPassword123",
        )
        other_user = await crud_user.create(db_session, obj_in=other_user_data)
        other_user.status = UserStatus.ACTIVE
        other_user.is_verified = True
        other_user.role = UserRole.CREATOR
        db_session.add(other_user)
        await db_session.commit()
        
        # Login as other user
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={"username": other_user.email, "password": "TestPassword123"},
        )
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Act
        response = await async_client.put(
            f"/api/v1/videos/{video.id}",
            headers=headers,
            json={"title": "Hacked Title"},
        )
        
        # Assert
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_search_with_empty_results(
        self,
        async_client: AsyncClient,
    ):
        """Test search with no matching results."""
        # Act
        response = await async_client.get("/api/v1/videos/search?q=nonexistent12345xyz")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["items"]) == 0
