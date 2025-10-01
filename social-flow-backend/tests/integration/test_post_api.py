"""
Integration tests for Post API endpoints.

This module contains integration tests for the post endpoints
including CRUD operations, feed generation, and interactions.
"""

import pytest
from httpx import AsyncClient
from app.models import User, Post


class TestPostEndpoints:
    """Integration tests for post endpoints."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_post_success(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test creating a post via API."""
        post_data = {
            "content": "This is a test post #test @user",
            "visibility": "public",
        }
        
        response = await async_client.post(
            "/api/v1/posts/",
            json=post_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["content"] == post_data["content"]
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_post_unauthorized(self, async_client: AsyncClient):
        """Test creating a post without authentication."""
        post_data = {
            "content": "Test post",
            "visibility": "public",
        }
        
        response = await async_client.post(
            "/api/v1/posts/",
            json=post_data,
        )
        
        assert response.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_post_by_id(self, async_client: AsyncClient, test_post: Post):
        """Test getting a specific post."""
        response = await async_client.get(f"/api/v1/posts/{test_post.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_post.id
        assert data["content"] == test_post.content

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_post_not_found(self, async_client: AsyncClient):
        """Test getting a non-existent post."""
        response = await async_client.get("/api/v1/posts/nonexistent_id")
        
        assert response.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_post_success(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test updating a post."""
        update_data = {
            "content": "Updated content",
        }
        
        response = await async_client.patch(
            f"/api/v1/posts/{test_post.id}",
            json=update_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == update_data["content"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_update_post_unauthorized(self, async_client: AsyncClient, test_post: Post):
        """Test updating a post without proper authorization."""
        update_data = {
            "content": "Updated content",
        }
        
        # Create different user headers
        different_auth_headers = {"Authorization": "Bearer different_user_token"}
        
        response = await async_client.patch(
            f"/api/v1/posts/{test_post.id}",
            json=update_data,
            headers=different_auth_headers,
        )
        
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_delete_post_success(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test deleting a post."""
        response = await async_client.delete(
            f"/api/v1/posts/{test_post.id}",
            headers=auth_headers,
        )
        
        assert response.status_code == 204

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_like_post_success(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test liking a post."""
        response = await async_client.post(
            f"/api/v1/posts/{test_post.id}/like",
            headers=auth_headers,
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "like_count" in data or "status" in data

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_unlike_post_success(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test unliking a post."""
        # First like the post
        await async_client.post(
            f"/api/v1/posts/{test_post.id}/like",
            headers=auth_headers,
        )
        
        # Then unlike it
        response = await async_client.delete(
            f"/api/v1/posts/{test_post.id}/like",
            headers=auth_headers,
        )
        
        assert response.status_code in [200, 204]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_user_feed(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test getting personalized user feed."""
        response = await async_client.get(
            "/api/v1/posts/feed",
            headers=auth_headers,
            params={"algorithm": "chronological", "limit": 10},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_trending_posts(self, async_client: AsyncClient):
        """Test getting trending posts."""
        response = await async_client.get(
            "/api/v1/posts/trending",
            params={"limit": 10},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_posts_by_hashtag(self, async_client: AsyncClient):
        """Test searching posts by hashtag."""
        response = await async_client.get(
            "/api/v1/posts/search",
            params={"hashtag": "test", "limit": 10},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_user_posts(self, async_client: AsyncClient, test_user: User):
        """Test getting all posts by a specific user."""
        response = await async_client.get(
            f"/api/v1/users/{test_user.id}/posts",
            params={"limit": 10},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # All posts should belong to the user
        for post in data:
            assert post["owner_id"] == test_user.id

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_comment_on_post(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test creating a comment on a post."""
        comment_data = {
            "content": "This is a test comment",
        }
        
        response = await async_client.post(
            f"/api/v1/posts/{test_post.id}/comments",
            json=comment_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["content"] == comment_data["content"]
        assert "id" in data

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_post_comments(self, async_client: AsyncClient, test_post: Post):
        """Test getting all comments on a post."""
        response = await async_client.get(
            f"/api/v1/posts/{test_post.id}/comments",
            params={"limit": 10},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_share_post(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test sharing a post."""
        response = await async_client.post(
            f"/api/v1/posts/{test_post.id}/share",
            headers=auth_headers,
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "share_count" in data or "status" in data

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_report_post(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test reporting a post."""
        report_data = {
            "reason": "spam",
            "description": "This post contains spam content",
        }
        
        response = await async_client.post(
            f"/api/v1/posts/{test_post.id}/report",
            json=report_data,
            headers=auth_headers,
        )
        
        assert response.status_code in [200, 201]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_get_post_analytics(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test getting post analytics."""
        response = await async_client.get(
            f"/api/v1/posts/{test_post.id}/analytics",
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "views" in data
        assert "likes" in data
        assert "comments" in data

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_feed_pagination(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test feed pagination."""
        # Get first page
        response1 = await async_client.get(
            "/api/v1/posts/feed",
            headers=auth_headers,
            params={"limit": 5, "offset": 0},
        )
        
        assert response1.status_code == 200
        page1 = response1.json()
        
        # Get second page
        response2 = await async_client.get(
            "/api/v1/posts/feed",
            headers=auth_headers,
            params={"limit": 5, "offset": 5},
        )
        
        assert response2.status_code == 200
        page2 = response2.json()
        
        # Pages should be different
        if len(page1) > 0 and len(page2) > 0:
            assert page1[0]["id"] != page2[0]["id"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_post_with_media(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test creating a post with media attachments."""
        post_data = {
            "content": "Post with media",
            "visibility": "public",
            "media_urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
        }
        
        response = await async_client.post(
            "/api/v1/posts/",
            json=post_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 201
        data = response.json()
        if "media_urls" in data:
            assert len(data["media_urls"]) == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_post_validation_error(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test post creation with invalid data."""
        post_data = {
            "content": "",  # Empty content should fail validation
            "visibility": "public",
        }
        
        response = await async_client.post(
            "/api/v1/posts/",
            json=post_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_double_like_prevention(self, async_client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict):
        """Test that a user cannot like the same post twice."""
        # First like
        response1 = await async_client.post(
            f"/api/v1/posts/{test_post.id}/like",
            headers=auth_headers,
        )
        assert response1.status_code in [200, 201]
        
        # Second like attempt
        response2 = await async_client.post(
            f"/api/v1/posts/{test_post.id}/like",
            headers=auth_headers,
        )
        # Should either return 200 (idempotent) or 400 (already liked)
        assert response2.status_code in [200, 400]
