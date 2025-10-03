"""
Integration tests for Post API endpoints (Clean Architecture).

Tests post creation, replies, engagement, feeds, and discovery.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient


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


class TestPostCreation:
    """Test post creation endpoint."""
    
    @pytest.mark.asyncio
    async def test_create_post_success(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test successful post creation."""
        response = await async_client.post(
            "/api/v1/v2/posts/",
            json={
                "content": "This is my first post!",
                "visibility": "public",
            },
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "This is my first post!"
        assert data["visibility"] == "public"
        assert data["author_id"] == registered_user["id"]
        assert "id" in data
        assert "created_at" in data
    
    @pytest.mark.asyncio
    async def test_create_post_with_media(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test creating post with media attachments."""
        response = await async_client.post(
            "/api/v1/v2/posts/",
            json={
                "content": "Check out this photo!",
                "visibility": "public",
                "media_urls": ["https://example.com/image.jpg"],
            },
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        assert response.status_code == 201
        data = response.json()
        assert len(data["media_urls"]) == 1
    
    @pytest.mark.asyncio
    async def test_create_post_without_auth(self, async_client: AsyncClient):
        """Test creating post without authentication."""
        response = await async_client.post(
            "/api/v1/v2/posts/",
            json={"content": "This should fail"},
        )
        
        # Should require authentication
        assert response.status_code in [401, 403]
    
    @pytest.mark.asyncio
    async def test_create_empty_post(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test creating post with empty content."""
        response = await async_client.post(
            "/api/v1/v2/posts/",
            json={"content": "", "visibility": "public"},
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should require content
        assert response.status_code == 422


class TestPostRetrieval:
    """Test post retrieval endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_post_by_id(self, async_client: AsyncClient):
        """Test getting post by ID."""
        # Test 404 for nonexistent post
        response = await async_client.get(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000"
        )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_user_posts(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test getting user's posts."""
        response = await async_client.get(
            f"/api/v1/v2/posts/user/{registered_user['id']}?skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestPostReplies:
    """Test post reply functionality."""
    
    @pytest.mark.asyncio
    async def test_create_reply(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test creating a reply to a post."""
        # Would need to create a post first
        # Test with mock post ID
        response = await async_client.post(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000/replies",
            json={"content": "This is a reply"},
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent post
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_post_replies(self, async_client: AsyncClient):
        """Test getting replies for a post."""
        response = await async_client.get(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000/replies?skip=0&limit=20"
        )
        
        # Should return 404 for nonexistent post
        assert response.status_code == 404


class TestPostEngagement:
    """Test post engagement endpoints."""
    
    @pytest.mark.asyncio
    async def test_like_post(self, async_client: AsyncClient, registered_user: dict):
        """Test liking a post."""
        response = await async_client.post(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000/like",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent post
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_unlike_post(self, async_client: AsyncClient, registered_user: dict):
        """Test unliking a post."""
        response = await async_client.delete(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000/like",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent post
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_share_post(self, async_client: AsyncClient, registered_user: dict):
        """Test sharing/reposting a post."""
        response = await async_client.post(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000/share",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent post
        assert response.status_code == 404


class TestPostFeeds:
    """Test post feed endpoints."""
    
    @pytest.mark.asyncio
    async def test_personalized_feed(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test getting personalized feed."""
        response = await async_client.get(
            "/api/v1/v2/posts/feed/personalized?skip=0&limit=20",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_global_feed(self, async_client: AsyncClient):
        """Test getting global feed."""
        response = await async_client.get(
            "/api/v1/v2/posts/feed/global?skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_following_feed(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test getting feed from followed users."""
        response = await async_client.get(
            "/api/v1/v2/posts/feed/following?skip=0&limit=20",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestPostSearch:
    """Test post search and discovery."""
    
    @pytest.mark.asyncio
    async def test_search_posts(self, async_client: AsyncClient):
        """Test searching posts."""
        response = await async_client.get(
            "/api/v1/v2/posts/search?query=test&skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_trending_posts(self, async_client: AsyncClient):
        """Test getting trending posts."""
        response = await async_client.get(
            "/api/v1/v2/posts/trending?days=7&skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_hashtag_posts(self, async_client: AsyncClient):
        """Test getting posts by hashtag."""
        response = await async_client.get(
            "/api/v1/v2/posts/hashtag/test?skip=0&limit=20"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestPostManagement:
    """Test post management endpoints."""
    
    @pytest.mark.asyncio
    async def test_update_post(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test updating a post."""
        response = await async_client.put(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000",
            json={"content": "Updated content"},
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent post
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_delete_post(
        self,
        async_client: AsyncClient,
        registered_user: dict,
    ):
        """Test deleting a post."""
        response = await async_client.delete(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {registered_user['token']}"},
        )
        
        # Should return 404 for nonexistent post
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_update_post_without_auth(self, async_client: AsyncClient):
        """Test updating post without authentication."""
        response = await async_client.put(
            "/api/v1/v2/posts/00000000-0000-0000-0000-000000000000",
            json={"content": "Updated content"},
        )
        
        # Should require authentication
        assert response.status_code in [401, 403]
