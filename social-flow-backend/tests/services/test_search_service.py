"""
Tests for Search Service.

This module contains comprehensive tests for the smart search functionality.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from app.services.search_service import SearchService
from app.models.video import Video, VideoStatus, VideoVisibility
from app.models.social import Post
from app.auth.models.user import User


@pytest.mark.asyncio
class TestSearchService:
    """Test suite for SearchService."""
    
    async def test_search_videos_by_title(self, db_session):
        """Test video search by title."""
        # Create test videos
        user = User(
            id=uuid4(),
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        video1 = Video(
            id=uuid4(),
            title="Python Programming Tutorial",
            description="Learn Python",
            filename="test.mp4",
            file_size=1000,
            s3_key="test/test.mp4",
            s3_bucket="test-bucket",
            owner_id=user.id,
            visibility=VideoVisibility.PUBLIC,
            status=VideoStatus.PROCESSED,
            is_approved=True,
            views_count=100,
        )
        
        video2 = Video(
            id=uuid4(),
            title="JavaScript Basics",
            description="Learn JavaScript",
            filename="test2.mp4",
            file_size=1000,
            s3_key="test/test2.mp4",
            s3_bucket="test-bucket",
            owner_id=user.id,
            visibility=VideoVisibility.PUBLIC,
            status=VideoStatus.PROCESSED,
            is_approved=True,
            views_count=50,
        )
        
        db_session.add(video1)
        db_session.add(video2)
        await db_session.commit()
        
        # Test search
        search_service = SearchService(db_session)
        result = await search_service.search_videos(
            query="Python",
            limit=10,
            offset=0,
        )
        
        assert result["total"] >= 1
        assert len(result["results"]) >= 1
        assert result["results"][0]["title"] == "Python Programming Tutorial"
    
    async def test_search_videos_with_filters(self, db_session):
        """Test video search with duration filters."""
        user = User(
            id=uuid4(),
            username="testuser2",
            email="test2@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        # Short video
        video_short = Video(
            id=uuid4(),
            title="Short Video",
            description="Brief tutorial",
            filename="short.mp4",
            file_size=500,
            duration=120.0,  # 2 minutes
            s3_key="test/short.mp4",
            s3_bucket="test-bucket",
            owner_id=user.id,
            visibility=VideoVisibility.PUBLIC,
            status=VideoStatus.PROCESSED,
            is_approved=True,
        )
        
        # Long video
        video_long = Video(
            id=uuid4(),
            title="Long Video",
            description="Detailed tutorial",
            filename="long.mp4",
            file_size=2000,
            duration=3600.0,  # 1 hour
            s3_key="test/long.mp4",
            s3_bucket="test-bucket",
            owner_id=user.id,
            visibility=VideoVisibility.PUBLIC,
            status=VideoStatus.PROCESSED,
            is_approved=True,
        )
        
        db_session.add(video_short)
        db_session.add(video_long)
        await db_session.commit()
        
        # Search with duration filter
        search_service = SearchService(db_session)
        result = await search_service.search_videos(
            query="Video",
            filters={"duration_min": 1000},
            limit=10,
        )
        
        # Should only return long video
        assert all(r["duration"] >= 1000 for r in result["results"])
    
    async def test_search_posts_by_content(self, db_session):
        """Test post search by content."""
        user = User(
            id=uuid4(),
            username="testuser3",
            email="test3@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        post = Post(
            id=uuid4(),
            content="This is a test post about #python programming",
            owner_id=user.id,
            is_approved=True,
            is_flagged=False,
            likes_count=10,
        )
        
        db_session.add(post)
        await db_session.commit()
        
        # Test search
        search_service = SearchService(db_session)
        result = await search_service.search_posts(
            query="python",
            limit=10,
        )
        
        assert result["total"] >= 1
        assert len(result["results"]) >= 1
        assert "python" in result["results"][0]["content"].lower()
    
    async def test_search_users(self, db_session):
        """Test user search by username."""
        user = User(
            id=uuid4(),
            username="johndoe",
            email="john@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        await db_session.commit()
        
        # Test search
        search_service = SearchService(db_session)
        result = await search_service.search_users(
            query="john",
            limit=10,
        )
        
        assert result["total"] >= 1
        assert len(result["results"]) >= 1
        assert result["results"][0]["username"] == "johndoe"
    
    async def test_get_suggestions(self, db_session):
        """Test search suggestions."""
        user = User(
            id=uuid4(),
            username="testuser4",
            email="test4@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        # Create videos for suggestions
        for i in range(3):
            video = Video(
                id=uuid4(),
                title=f"Python Tutorial Part {i+1}",
                description=f"Learn Python - Part {i+1}",
                filename=f"python{i}.mp4",
                file_size=1000,
                s3_key=f"test/python{i}.mp4",
                s3_bucket="test-bucket",
                owner_id=user.id,
                visibility=VideoVisibility.PUBLIC,
                status=VideoStatus.PROCESSED,
                is_approved=True,
                views_count=100 - i * 10,
            )
            db_session.add(video)
        
        await db_session.commit()
        
        # Test suggestions
        search_service = SearchService(db_session)
        result = await search_service.get_suggestions(
            query="Pyth",
            limit=10,
        )
        
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0
    
    async def test_search_ranking(self, db_session):
        """Test that search results are properly ranked."""
        user = User(
            id=uuid4(),
            username="testuser5",
            email="test5@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        # Popular video
        popular_video = Video(
            id=uuid4(),
            title="Programming Guide",
            description="Guide",
            filename="popular.mp4",
            file_size=1000,
            s3_key="test/popular.mp4",
            s3_bucket="test-bucket",
            owner_id=user.id,
            visibility=VideoVisibility.PUBLIC,
            status=VideoStatus.PROCESSED,
            is_approved=True,
            views_count=10000,
            likes_count=500,
        )
        
        # Less popular video
        unpopular_video = Video(
            id=uuid4(),
            title="Programming Basics",
            description="Basics",
            filename="unpopular.mp4",
            file_size=1000,
            s3_key="test/unpopular.mp4",
            s3_bucket="test-bucket",
            owner_id=user.id,
            visibility=VideoVisibility.PUBLIC,
            status=VideoStatus.PROCESSED,
            is_approved=True,
            views_count=10,
            likes_count=1,
        )
        
        db_session.add(popular_video)
        db_session.add(unpopular_video)
        await db_session.commit()
        
        # Test search with relevance sorting
        search_service = SearchService(db_session)
        result = await search_service.search_videos(
            query="Programming",
            sort_by="relevance",
            limit=10,
        )
        
        # Popular video should rank higher
        assert result["results"][0]["views_count"] > result["results"][1]["views_count"]
