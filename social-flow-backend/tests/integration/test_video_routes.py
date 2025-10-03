"""
Integration tests for Video API endpoints (Clean Architecture).

Tests video upload, management, engagement, and discovery workflows.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from io import BytesIO


@pytest_asyncio.fixture
async def registered_user(async_client: AsyncClient):
    """Create and return a registered user with auth token."""
    response = await async_client.post(
        "/api/v1/v2/users/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!",
            "display_name": "Test User",
        },
    )
    user = response.json()
    
    # Get auth token (mock for now)
    user["token"] = "mock_token"
    
    return user


class TestVideoUpload:
    """Test video upload endpoint."""
    
    @pytest.mark.asyncio
    async def test_upload_video_success(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test successful video upload."""
        # Create mock video file
        video_data = BytesIO(b"fake video data")
        video_data.name = "test_video.mp4"
        
        response = await async_client.post(
            "/api/v1/v2/videos/upload",
            data={
                "title": "Test Video",
                "description": "This is a test video",
                "tags": "test,video",
            },
            files={"file": ("test_video.mp4", video_data, "video/mp4")},
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "video_id" in data
        assert data["status"] == "processing"
    
    @pytest.mark.asyncio
    async def test_upload_video_without_auth(self, async_client: AsyncClient):
        """Test video upload without authentication."""
        video_data = BytesIO(b"fake video data")
        
        response = await async_client.post(
            "/api/v1/v2/videos/upload",
            data={"title": "Test Video"},
            files={"file": ("test_video.mp4", video_data, "video/mp4")},
        )
        
        # Should require authentication
        assert response.status_code in [401, 403]


class TestVideoRetrieval:
    """Test video retrieval endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_video_by_id(self, async_client: AsyncClient):
        """Test getting video by ID."""
        # This would require setting up a video first
        # For now, test 404 for nonexistent video
        response = await async_client.get(
            "/api/v1/v2/videos/00000000-0000-0000-0000-000000000000"
        )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_discover_videos(self, async_client: AsyncClient):
        """Test discovery feed."""
        response = await async_client.get("/api/v1/v2/videos/?skip=0&limit=20")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_discover_videos_pagination(self, async_client: AsyncClient):
        """Test discovery feed with pagination."""
        response = await async_client.get("/api/v1/v2/videos/?skip=10&limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5


class TestVideoEngagement:
    """Test video engagement endpoints."""
    
    @pytest.mark.asyncio
    async def test_like_video(self, async_client: AsyncClient, registered_user: dict):
        """Test liking a video."""
        # Would need to create a video first
        # Test with mock video ID
        response = await async_client.post(
            "/api/v1/v2/videos/00000000-0000-0000-0000-000000000000/like",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent video
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_record_view(self, async_client: AsyncClient):
        """Test recording video view."""
        response = await async_client.post(
            "/api/v1/v2/videos/00000000-0000-0000-0000-000000000000/view"
        )
        
        # Should return 404 for nonexistent video
        assert response.status_code == 404


class TestVideoSearch:
    """Test video search and filtering."""
    
    @pytest.mark.asyncio
    async def test_search_videos(self, async_client: AsyncClient):
        """Test video search."""
        response = await async_client.get(
            "/api/v1/v2/videos/search?query=test&skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_trending_videos(self, async_client: AsyncClient):
        """Test trending videos."""
        response = await async_client.get(
            "/api/v1/v2/videos/trending?days=7&skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_user_videos(self, async_client: AsyncClient, registered_user: dict):
        """Test getting user's videos."""
        response = await async_client.get(
            f"/api/v1/v2/videos/user/{registered_user['id']}?skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestVideoManagement:
    """Test video management endpoints."""
    
    @pytest.mark.asyncio
    async def test_update_video(self, async_client: AsyncClient, registered_user: dict):
        """Test updating video details."""
        response = await async_client.put(
            "/api/v1/v2/videos/00000000-0000-0000-0000-000000000000",
            json={"title": "Updated Title", "description": "Updated description"},
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent video
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_delete_video(self, async_client: AsyncClient, registered_user: dict):
        """Test deleting video."""
        response = await async_client.delete(
            "/api/v1/v2/videos/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent video
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_update_video_without_auth(self, async_client: AsyncClient):
        """Test updating video without authentication."""
        response = await async_client.put(
            "/api/v1/v2/videos/00000000-0000-0000-0000-000000000000",
            json={"title": "Updated Title"},
        )
        
        # Should require authentication
        assert response.status_code in [401, 403]
