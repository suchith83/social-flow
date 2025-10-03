"""
Unit tests for post service functionality.

This module contains unit tests for the post service
and feed generation logic.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from app.posts.services.post_service import PostService
from app.posts.schemas.post import PostCreate, PostUpdate
from app.models import Post, User


class TestPostService:
    """Test cases for PostService."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = AsyncMock()
        db.add = Mock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def post_service(self, mock_db):
        """Create PostService instance for testing."""
        service = PostService(mock_db)
        return service

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            id="user123",
            username="testuser",
            email="test@example.com",
            display_name="Test User",
        )

    @pytest.fixture
    def test_post(self, test_user):
        """Create test post."""
        return Post(
            id="post123",
            content="Test post content #hashtag @mention",
            owner_id=test_user.id,
            owner=test_user,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            view_count=100,
            like_count=10,
            comment_count=5,
            share_count=2,
        )

    @pytest.mark.asyncio
    async def test_create_post_success(self, post_service, mock_db, test_user):
        """Test successful post creation."""
        post_data = PostCreate(
            content="Test post content #test @user",
            visibility="public",
        )
        
        # Mock the result
        mock_post = Post(
            id="post123",
            content=post_data.content,
            owner_id=test_user.id,
            visibility=post_data.visibility,
        )
        
        with patch.object(post_service, '_extract_hashtags', return_value=["test"]):
            with patch.object(post_service, '_extract_mentions', return_value=["user"]):
                mock_db.refresh.side_effect = lambda obj: setattr(obj, 'id', 'post123')
                
                result = await post_service.create_post(post_data, test_user.id)
                
                mock_db.add.assert_called_once()
                mock_db.commit.assert_called_once()
                assert mock_db.refresh.called

    @pytest.mark.asyncio
    async def test_create_post_with_hashtags(self, post_service, test_user):
        """Test post creation with hashtag extraction."""
        content = "Test post #python #coding #ai"
        
        hashtags = post_service._extract_hashtags(content)
        
        assert len(hashtags) == 3
        assert "python" in hashtags
        assert "coding" in hashtags
        assert "ai" in hashtags

    @pytest.mark.asyncio
    async def test_create_post_with_mentions(self, post_service):
        """Test post creation with mention extraction."""
        content = "Test post @user1 @user2 @user3"
        
        mentions = post_service._extract_mentions(content)
        
        assert len(mentions) == 3
        assert "user1" in mentions
        assert "user2" in mentions
        assert "user3" in mentions

    @pytest.mark.asyncio
    async def test_update_post_success(self, post_service, mock_db, test_post, test_user):
        """Test successful post update."""
        update_data = PostUpdate(
            content="Updated post content",
        )
        
        with patch.object(post_service, 'get_post_by_id', return_value=test_post):
            result = await post_service.update_post(
                test_post.id,
                update_data,
                test_user.id
            )
            
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_post_unauthorized(self, post_service, test_post):
        """Test post update by unauthorized user."""
        update_data = PostUpdate(content="Updated content")
        unauthorized_user_id = "different_user"
        
        with patch.object(post_service, 'get_post_by_id', return_value=test_post):
            with pytest.raises(Exception):  # Should raise unauthorized exception
                await post_service.update_post(
                    test_post.id,
                    update_data,
                    unauthorized_user_id
                )

    @pytest.mark.asyncio
    async def test_delete_post_success(self, post_service, mock_db, test_post, test_user):
        """Test successful post deletion."""
        with patch.object(post_service, 'get_post_by_id', return_value=test_post):
            await post_service.delete_post(test_post.id, test_user.id)
            
            mock_db.delete.assert_called_once_with(test_post)
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_feed_chronological(self, post_service, mock_db, test_user):
        """Test getting user feed with chronological algorithm."""
        mock_posts = [
            Post(id=f"post{i}", content=f"Post {i}", owner_id=test_user.id)
            for i in range(10)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_posts
        mock_db.execute.return_value = mock_result
        
        result = await post_service.get_user_feed(
            test_user.id,
            algorithm="chronological",
            limit=10
        )
        
        assert len(result) == 10
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_feed_engagement(self, post_service, mock_db, test_user):
        """Test getting user feed with engagement algorithm."""
        mock_posts = [
            Post(
                id=f"post{i}",
                content=f"Post {i}",
                owner_id=test_user.id,
                like_count=100 - i * 10,
                comment_count=50 - i * 5,
            )
            for i in range(10)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_posts
        mock_db.execute.return_value = mock_result
        
        result = await post_service.get_user_feed(
            test_user.id,
            algorithm="engagement",
            limit=10
        )
        
        assert len(result) == 10
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_trending_posts(self, post_service, mock_db):
        """Test getting trending posts."""
        mock_posts = [
            Post(
                id=f"post{i}",
                content=f"Trending post {i}",
                like_count=1000 - i * 100,
                comment_count=500 - i * 50,
                share_count=100 - i * 10,
                view_count=10000 - i * 1000,
            )
            for i in range(10)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_posts
        mock_db.execute.return_value = mock_result
        
        result = await post_service.get_trending_posts(limit=10)
        
        assert len(result) == 10
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_like_post_success(self, post_service, mock_db, test_post, test_user):
        """Test liking a post."""
        with patch.object(post_service, 'get_post_by_id', return_value=test_post):
            with patch.object(post_service, '_check_existing_like', return_value=None):
                result = await post_service.like_post(test_post.id, test_user.id)
                
                mock_db.add.assert_called_once()
                mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_unlike_post_success(self, post_service, mock_db, test_post, test_user):
        """Test unliking a post."""
        mock_like = Mock()
        
        with patch.object(post_service, 'get_post_by_id', return_value=test_post):
            with patch.object(post_service, '_check_existing_like', return_value=mock_like):
                await post_service.unlike_post(test_post.id, test_user.id)
                
                mock_db.delete.assert_called_once_with(mock_like)
                mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_get_post_analytics(self, post_service, mock_db, test_post):
        """Test getting post analytics."""
        with patch.object(post_service, 'get_post_by_id', return_value=test_post):
            result = await post_service.get_post_analytics(test_post.id)
            
            assert "views" in result
            assert "likes" in result
            assert "comments" in result
            assert "shares" in result
            assert result["views"] == test_post.view_count
            assert result["likes"] == test_post.like_count

    @pytest.mark.asyncio
    async def test_search_posts_by_hashtag(self, post_service, mock_db):
        """Test searching posts by hashtag."""
        hashtag = "python"
        mock_posts = [
            Post(id=f"post{i}", content=f"Post {i} #python")
            for i in range(5)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_posts
        mock_db.execute.return_value = mock_result
        
        result = await post_service.search_posts_by_hashtag(hashtag)
        
        assert len(result) == 5
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_posts(self, post_service, mock_db, test_user):
        """Test getting all posts by a user."""
        mock_posts = [
            Post(id=f"post{i}", content=f"Post {i}", owner_id=test_user.id)
            for i in range(15)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_posts[:10]
        mock_db.execute.return_value = mock_result
        
        result = await post_service.get_user_posts(test_user.id, limit=10)
        
        assert len(result) <= 10
        mock_db.execute.assert_called_once()

    def test_extract_hashtags_edge_cases(self, post_service):
        """Test hashtag extraction with edge cases."""
        # No hashtags
        assert post_service._extract_hashtags("No hashtags here") == []
        
        # Multiple hashtags
        content = "#one #two #three"
        result = post_service._extract_hashtags(content)
        assert len(result) == 3
        
        # Duplicate hashtags
        content = "#test #test #test"
        result = post_service._extract_hashtags(content)
        assert len(result) == 1
        
        # Mixed case
        content = "#Test #TEST #test"
        result = post_service._extract_hashtags(content)
        assert all(tag.islower() for tag in result)

    def test_extract_mentions_edge_cases(self, post_service):
        """Test mention extraction with edge cases."""
        # No mentions
        assert post_service._extract_mentions("No mentions here") == []
        
        # Multiple mentions
        content = "@user1 @user2 @user3"
        result = post_service._extract_mentions(content)
        assert len(result) == 3
        
        # Duplicate mentions
        content = "@user @user @user"
        result = post_service._extract_mentions(content)
        assert len(result) == 1
        
        # Email addresses should not be extracted as mentions
        content = "Contact me at test@example.com"
        result = post_service._extract_mentions(content)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_increment_view_count(self, post_service, mock_db, test_post):
        """Test incrementing post view count."""
        initial_count = test_post.view_count
        
        with patch.object(post_service, 'get_post_by_id', return_value=test_post):
            await post_service.increment_view_count(test_post.id)
            
            mock_db.commit.assert_called_once()
            # View count should be incremented
            assert test_post.view_count >= initial_count
