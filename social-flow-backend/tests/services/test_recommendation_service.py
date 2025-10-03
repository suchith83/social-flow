"""
Tests for Recommendation Service.

This module contains comprehensive tests for the recommendation engine.
"""

import pytest
from uuid import uuid4

from app.services.recommendation_service import RecommendationService
from app.models.video import Video, VideoStatus, VideoVisibility
from app.models.social import Post
from app.auth.models.user import User
from app.models.social import Follow


@pytest.mark.asyncio
class TestRecommendationService:
    """Test suite for RecommendationService."""
    
    async def test_get_video_recommendations_anonymous(self, db_session):
        """Test video recommendations for anonymous users."""
        user = User(
            id=uuid4(),
            username="creator",
            email="creator@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        # Create trending videos
        for i in range(5):
            video = Video(
                id=uuid4(),
                title=f"Trending Video {i+1}",
                description=f"Popular content {i+1}",
                filename=f"trending{i}.mp4",
                file_size=1000,
                s3_key=f"test/trending{i}.mp4",
                s3_bucket="test-bucket",
                owner_id=user.id,
                visibility=VideoVisibility.PUBLIC,
                status=VideoStatus.PROCESSED,
                is_approved=True,
                views_count=1000 + i * 100,
                likes_count=50 + i * 10,
            )
            db_session.add(video)
        
        await db_session.commit()
        
        # Test recommendations for anonymous user
        rec_service = RecommendationService(db_session)
        result = await rec_service.get_video_recommendations(
            user_id=None,
            limit=3,
            algorithm="trending",
        )
        
        assert result["count"] == 3
        assert result["algorithm"] == "trending"
        assert len(result["recommendations"]) == 3
    
    async def test_get_video_recommendations_authenticated(self, db_session):
        """Test personalized video recommendations."""
        # Create users
        creator = User(
            id=uuid4(),
            username="creator2",
            email="creator2@example.com",
            hashed_password="hash",
            is_active=True,
        )
        
        viewer = User(
            id=uuid4(),
            username="viewer",
            email="viewer@example.com",
            hashed_password="hash",
            is_active=True,
        )
        
        db_session.add(creator)
        db_session.add(viewer)
        
        # Viewer follows creator
        follow = Follow(
            follower_id=viewer.id,
            following_id=creator.id,
        )
        db_session.add(follow)
        
        # Create videos from followed creator
        for i in range(3):
            video = Video(
                id=uuid4(),
                title=f"Creator Video {i+1}",
                description=f"Content from followed creator {i+1}",
                filename=f"creator{i}.mp4",
                file_size=1000,
                s3_key=f"test/creator{i}.mp4",
                s3_bucket="test-bucket",
                owner_id=creator.id,
                visibility=VideoVisibility.PUBLIC,
                status=VideoStatus.PROCESSED,
                is_approved=True,
                views_count=500,
                likes_count=25,
            )
            db_session.add(video)
        
        await db_session.commit()
        
        # Test recommendations
        rec_service = RecommendationService(db_session)
        result = await rec_service.get_video_recommendations(
            user_id=viewer.id,
            limit=5,
            algorithm="collaborative",
        )
        
        assert result["count"] >= 1
        assert len(result["recommendations"]) >= 1
    
    async def test_get_feed_recommendations(self, db_session):
        """Test feed recommendations."""
        # Create users
        creator = User(
            id=uuid4(),
            username="creator3",
            email="creator3@example.com",
            hashed_password="hash",
            is_active=True,
        )
        
        viewer = User(
            id=uuid4(),
            username="viewer2",
            email="viewer2@example.com",
            hashed_password="hash",
            is_active=True,
        )
        
        db_session.add(creator)
        db_session.add(viewer)
        
        # Viewer follows creator
        follow = Follow(
            follower_id=viewer.id,
            following_id=creator.id,
        )
        db_session.add(follow)
        
        # Create posts from followed creator
        for i in range(5):
            post = Post(
                id=uuid4(),
                content=f"Post from followed creator #{i+1}",
                owner_id=creator.id,
                is_approved=True,
                is_flagged=False,
                likes_count=10 + i,
            )
            db_session.add(post)
        
        await db_session.commit()
        
        # Test feed recommendations
        rec_service = RecommendationService(db_session)
        result = await rec_service.get_feed_recommendations(
            user_id=viewer.id,
            limit=10,
            algorithm="following",
        )
        
        assert result["count"] >= 1
        assert len(result["recommendations"]) >= 1
    
    async def test_hybrid_recommendations(self, db_session):
        """Test hybrid recommendation algorithm."""
        user = User(
            id=uuid4(),
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        # Create diverse videos
        video_types = [
            ("Tutorial", 1000, 100),  # Popular
            ("Review", 500, 50),      # Medium
            ("Vlog", 100, 10),        # Less popular
        ]
        
        for i, (vtype, views, likes) in enumerate(video_types):
            video = Video(
                id=uuid4(),
                title=f"{vtype} Video {i+1}",
                description=f"{vtype} content",
                filename=f"{vtype.lower()}{i}.mp4",
                file_size=1000,
                s3_key=f"test/{vtype.lower()}{i}.mp4",
                s3_bucket="test-bucket",
                owner_id=user.id,
                visibility=VideoVisibility.PUBLIC,
                status=VideoStatus.PROCESSED,
                is_approved=True,
                views_count=views,
                likes_count=likes,
            )
            db_session.add(video)
        
        await db_session.commit()
        
        # Test hybrid recommendations
        rec_service = RecommendationService(db_session)
        result = await rec_service.get_video_recommendations(
            user_id=user.id,
            limit=10,
            algorithm="hybrid",
        )
        
        # Should include mix of different types
        assert result["count"] >= 1
        assert result["algorithm"] == "hybrid"
    
    async def test_recommendation_diversity(self, db_session):
        """Test that recommendations are diverse (not all from same creator)."""
        # Create multiple creators
        creators = []
        for i in range(3):
            creator = User(
                id=uuid4(),
                username=f"creator{i}",
                email=f"creator{i}@example.com",
                hashed_password="hash",
                is_active=True,
            )
            creators.append(creator)
            db_session.add(creator)
        
        # Each creator has multiple videos
        for creator in creators:
            for j in range(5):
                video = Video(
                    id=uuid4(),
                    title=f"Video by {creator.username} #{j+1}",
                    description="Content",
                    filename=f"video{j}.mp4",
                    file_size=1000,
                    s3_key=f"test/video{j}.mp4",
                    s3_bucket="test-bucket",
                    owner_id=creator.id,
                    visibility=VideoVisibility.PUBLIC,
                    status=VideoStatus.PROCESSED,
                    is_approved=True,
                    views_count=1000,
                    likes_count=50,
                )
                db_session.add(video)
        
        await db_session.commit()
        
        # Test recommendations
        rec_service = RecommendationService(db_session)
        result = await rec_service.get_video_recommendations(
            user_id=None,
            limit=6,
            algorithm="hybrid",
        )
        
        # Check diversity - should have videos from different creators
        owner_ids = set(r["owner_id"] for r in result["recommendations"])
        assert len(owner_ids) > 1, "Recommendations should include multiple creators"
    
    async def test_trending_posts(self, db_session):
        """Test trending post recommendations."""
        user = User(
            id=uuid4(),
            username="poster",
            email="poster@example.com",
            hashed_password="hash",
            is_active=True,
        )
        db_session.add(user)
        
        # Create posts with varying engagement
        for i in range(5):
            post = Post(
                id=uuid4(),
                content=f"Trending post #{i+1}",
                owner_id=user.id,
                is_approved=True,
                is_flagged=False,
                likes_count=100 + i * 20,
                reposts_count=10 + i * 5,
                comments_count=20 + i * 10,
            )
            db_session.add(post)
        
        await db_session.commit()
        
        # Test trending recommendations
        rec_service = RecommendationService(db_session)
        result = await rec_service.get_feed_recommendations(
            user_id=user.id,
            limit=5,
            algorithm="trending",
        )
        
        assert result["count"] >= 1
        # Most engaged post should be first
        if len(result["recommendations"]) > 1:
            first_engagement = (
                result["recommendations"][0]["likes_count"] +
                result["recommendations"][0]["reposts_count"] * 3
            )
            second_engagement = (
                result["recommendations"][1]["likes_count"] +
                result["recommendations"][1]["reposts_count"] * 3
            )
            assert first_engagement >= second_engagement
