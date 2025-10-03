"""
Unit tests for ML service functionality.

This module contains unit tests for the ML service
including content moderation, recommendations, and trending analysis.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from app.ml.services.ml_service import MLService
from app.models import User, Video, Post


class TestMLService:
    """Test cases for MLService."""

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
    def ml_service(self, mock_db):
        """Create MLService instance for testing."""
        # MLService doesn't take any arguments
        return MLService()

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            id="user123",
            username="testuser",
            email="test@example.com",
        )

    @pytest.fixture
    def test_video(self, test_user):
        """Create test video."""
        return Video(
            id="video123",
            title="Test Video",
            description="Test description",
            s3_key="videos/test.mp4",
            owner_id=test_user.id,
        )

    @pytest.fixture
    def test_post(self, test_user):
        """Create test post."""
        return Post(
            id="post123",
            content="Test post content",
            owner_id=test_user.id,
        )

    @pytest.mark.asyncio
    async def test_moderate_text_safe(self, ml_service):
        """Test text moderation with safe content."""
        text = "This is a perfectly safe and friendly message."
        
        with patch.object(ml_service, '_analyze_text_content') as mock_analyze:
            mock_analyze.return_value = {
                "is_safe": True,
                "toxicity_score": 0.05,
                "categories": {},
                "flagged": False,
            }
            
            result = await ml_service.moderate_text(text)
            
            assert result["is_safe"] is True
            assert result["toxicity_score"] < 0.5
            assert result["flagged"] is False
            mock_analyze.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_moderate_text_toxic(self, ml_service):
        """Test text moderation with toxic content."""
        text = "This text contains inappropriate content"
        
        with patch.object(ml_service, '_analyze_text_content') as mock_analyze:
            mock_analyze.return_value = {
                "is_safe": False,
                "toxicity_score": 0.95,
                "categories": {
                    "hate_speech": 0.9,
                    "harassment": 0.8,
                },
                "flagged": True,
            }
            
            result = await ml_service.moderate_text(text)
            
            assert result["is_safe"] is False
            assert result["toxicity_score"] > 0.5
            assert result["flagged"] is True
            assert "hate_speech" in result["categories"]

    @pytest.mark.asyncio
    async def test_moderate_image_safe(self, ml_service):
        """Test image moderation with safe content."""
        image_url = "https://example.com/safe-image.jpg"
        
        with patch.object(ml_service, '_analyze_image_content') as mock_analyze:
            mock_analyze.return_value = {
                "is_safe": True,
                "nsfw_score": 0.02,
                "categories": {
                    "safe": 0.98,
                },
                "flagged": False,
            }
            
            result = await ml_service.moderate_image(image_url)
            
            assert result["is_safe"] is True
            assert result["nsfw_score"] < 0.5
            assert result["flagged"] is False

    @pytest.mark.asyncio
    async def test_moderate_image_nsfw(self, ml_service):
        """Test image moderation with NSFW content."""
        image_url = "https://example.com/nsfw-image.jpg"
        
        with patch.object(ml_service, '_analyze_image_content') as mock_analyze:
            mock_analyze.return_value = {
                "is_safe": False,
                "nsfw_score": 0.92,
                "categories": {
                    "explicit": 0.9,
                },
                "flagged": True,
            }
            
            result = await ml_service.moderate_image(image_url)
            
            assert result["is_safe"] is False
            assert result["nsfw_score"] > 0.5
            assert result["flagged"] is True

    @pytest.mark.asyncio
    async def test_get_video_recommendations_for_user(self, ml_service, mock_db, test_user):
        """Test getting personalized video recommendations."""
        mock_videos = [
            Video(id=f"video{i}", title=f"Video {i}", owner_id=test_user.id)
            for i in range(10)
        ]
        
        with patch.object(ml_service, '_get_user_preferences', return_value={"categories": ["tech", "gaming"]}):
            with patch.object(ml_service, '_rank_videos_by_preferences') as mock_rank:
                mock_rank.return_value = mock_videos
                
                mock_result = MagicMock()
                mock_result.scalars.return_value.all.return_value = mock_videos
                mock_db.execute.return_value = mock_result
                
                result = await ml_service.get_video_recommendations(test_user.id, limit=10)
                
                assert len(result) <= 10
                mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_get_video_recommendations_cold_start(self, ml_service, mock_db):
        """Test video recommendations for new user (cold start)."""
        new_user_id = "new_user"
        mock_videos = [
            Video(id=f"video{i}", title=f"Popular Video {i}")
            for i in range(10)
        ]
        
        # Simulate cold start - no user preferences
        with patch.object(ml_service, '_get_user_preferences', return_value={}):
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = mock_videos
            mock_db.execute.return_value = mock_result
            
            result = await ml_service.get_video_recommendations(new_user_id, limit=10)
            
            # Should return popular videos for cold start
            assert len(result) <= 10

    @pytest.mark.asyncio
    async def test_analyze_trending_content(self, ml_service, mock_db):
        """Test trending content analysis."""
        mock_videos = [
            Video(
                id=f"video{i}",
                title=f"Trending Video {i}",
                view_count=10000 - i * 1000,
                like_count=1000 - i * 100,
            )
            for i in range(10)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_videos
        mock_db.execute.return_value = mock_result
        
        result = await ml_service.get_trending_videos(limit=10)
        
        assert len(result) <= 10
        # First video should have highest engagement
        if len(result) > 1:
            assert result[0].view_count >= result[1].view_count

    @pytest.mark.asyncio
    async def test_calculate_engagement_score(self, ml_service, test_video):
        """Test engagement score calculation."""
        test_video.view_count = 1000
        test_video.like_count = 100
        test_video.comment_count = 50
        test_video.share_count = 20
        
        score = ml_service.calculate_engagement_score(test_video)
        
        assert score > 0
        assert isinstance(score, (int, float))

    @pytest.mark.asyncio
    async def test_extract_video_features(self, ml_service, test_video):
        """Test video feature extraction."""
        with patch.object(ml_service, '_extract_video_embeddings') as mock_extract:
            mock_extract.return_value = {
                "visual_features": [0.1, 0.2, 0.3],
                "audio_features": [0.4, 0.5, 0.6],
                "text_features": [0.7, 0.8, 0.9],
            }
            
            result = await ml_service.extract_video_features(test_video.s3_key)
            
            assert "visual_features" in result
            assert "audio_features" in result
            assert "text_features" in result
            assert len(result["visual_features"]) > 0

    @pytest.mark.asyncio
    async def test_detect_spam_content_legitimate(self, ml_service):
        """Test spam detection with legitimate content."""
        content = "Check out this amazing video I just uploaded!"
        
        with patch.object(ml_service, '_analyze_spam_patterns') as mock_analyze:
            mock_analyze.return_value = {
                "is_spam": False,
                "spam_score": 0.15,
                "patterns_detected": [],
            }
            
            result = await ml_service.detect_spam(content)
            
            assert result["is_spam"] is False
            assert result["spam_score"] < 0.5

    @pytest.mark.asyncio
    async def test_detect_spam_content_spam(self, ml_service):
        """Test spam detection with spam content."""
        content = "BUY NOW!!! CLICK HERE!!! LIMITED TIME OFFER!!! www.spam.com"
        
        with patch.object(ml_service, '_analyze_spam_patterns') as mock_analyze:
            mock_analyze.return_value = {
                "is_spam": True,
                "spam_score": 0.95,
                "patterns_detected": ["excessive_caps", "suspicious_urls", "marketing_keywords"],
            }
            
            result = await ml_service.detect_spam(content)
            
            assert result["is_spam"] is True
            assert result["spam_score"] > 0.5
            assert len(result["patterns_detected"]) > 0

    @pytest.mark.asyncio
    async def test_generate_content_tags(self, ml_service, test_video):
        """Test automatic tag generation for content."""
        with patch.object(ml_service, '_extract_content_topics') as mock_extract:
            mock_extract.return_value = [
                {"tag": "technology", "confidence": 0.95},
                {"tag": "programming", "confidence": 0.89},
                {"tag": "tutorial", "confidence": 0.82},
            ]
            
            result = await ml_service.generate_tags(
                title=test_video.title,
                description=test_video.description
            )
            
            assert len(result) > 0
            assert all("tag" in item for item in result)
            assert all("confidence" in item for item in result)

    @pytest.mark.asyncio
    async def test_similarity_search(self, ml_service, mock_db, test_video):
        """Test finding similar content."""
        mock_similar_videos = [
            Video(id=f"video{i}", title=f"Similar Video {i}")
            for i in range(5)
        ]
        
        with patch.object(ml_service, '_compute_content_similarity') as mock_similarity:
            mock_similarity.return_value = mock_similar_videos
            
            result = await ml_service.find_similar_videos(test_video.id, limit=5)
            
            assert len(result) <= 5
            mock_similarity.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_preference_learning(self, ml_service, mock_db, test_user):
        """Test learning user preferences from interaction history."""
        mock_interactions = [
            {"video_id": f"video{i}", "action": "like", "category": "tech"}
            for i in range(10)
        ]
        
        with patch.object(ml_service, '_get_user_interactions', return_value=mock_interactions):
            preferences = await ml_service.learn_user_preferences(test_user.id)
            
            assert "categories" in preferences
            assert "keywords" in preferences or "categories" in preferences

    @pytest.mark.asyncio
    async def test_sentiment_analysis_positive(self, ml_service):
        """Test sentiment analysis with positive text."""
        text = "This is absolutely amazing! I love it so much!"
        
        with patch.object(ml_service, '_analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                "sentiment": "positive",
                "score": 0.92,
                "confidence": 0.95,
            }
            
            result = await ml_service.analyze_sentiment(text)
            
            assert result["sentiment"] == "positive"
            assert result["score"] > 0.5

    @pytest.mark.asyncio
    async def test_sentiment_analysis_negative(self, ml_service):
        """Test sentiment analysis with negative text."""
        text = "This is terrible and disappointing."
        
        with patch.object(ml_service, '_analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                "sentiment": "negative",
                "score": -0.85,
                "confidence": 0.90,
            }
            
            result = await ml_service.analyze_sentiment(text)
            
            assert result["sentiment"] == "negative"
            assert result["score"] < 0

    @pytest.mark.asyncio
    async def test_content_quality_score(self, ml_service, test_video):
        """Test content quality scoring."""
        test_video.view_count = 5000
        test_video.like_count = 450
        test_video.comment_count = 200
        test_video.share_count = 100
        
        with patch.object(ml_service, '_analyze_content_quality') as mock_analyze:
            mock_analyze.return_value = {
                "quality_score": 8.5,
                "engagement_rate": 0.15,
                "retention_rate": 0.75,
            }
            
            result = await ml_service.calculate_content_quality(test_video.id)
            
            assert result["quality_score"] > 0
            assert result["engagement_rate"] > 0

    @pytest.mark.asyncio
    async def test_predict_viral_potential(self, ml_service, test_video):
        """Test viral potential prediction."""
        with patch.object(ml_service, '_predict_virality') as mock_predict:
            mock_predict.return_value = {
                "viral_score": 0.78,
                "predicted_views": 50000,
                "confidence": 0.82,
                "factors": ["high_engagement", "trending_topic", "optimal_length"],
            }
            
            result = await ml_service.predict_viral_potential(test_video.id)
            
            assert result["viral_score"] >= 0
            assert result["viral_score"] <= 1
            assert "predicted_views" in result

    @pytest.mark.asyncio
    async def test_categorize_content(self, ml_service, test_video):
        """Test automatic content categorization."""
        with patch.object(ml_service, '_classify_content') as mock_classify:
            mock_classify.return_value = {
                "primary_category": "Education",
                "secondary_categories": ["Technology", "Tutorial"],
                "confidence": 0.89,
            }
            
            result = await ml_service.categorize_content(
                title=test_video.title,
                description=test_video.description
            )
            
            assert "primary_category" in result
            assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_detect_duplicate_content(self, ml_service, mock_db, test_video):
        """Test duplicate content detection."""
        with patch.object(ml_service, '_compute_content_hash') as mock_hash:
            mock_hash.return_value = "hash123"
            
            with patch.object(ml_service, '_find_similar_hashes') as mock_find:
                mock_find.return_value = []  # No duplicates found
                
                result = await ml_service.detect_duplicates(test_video.id)
                
                assert "is_duplicate" in result
                assert "similar_content" in result

    @pytest.mark.asyncio
    async def test_ml_cache_recommendations(self, ml_service, test_user):
        """Test caching of ML recommendations."""
        cache_key = f"recommendations:{test_user.id}"
        
        with patch.object(ml_service, '_get_from_cache') as mock_get_cache:
            mock_get_cache.return_value = None  # Cache miss
            
            with patch.object(ml_service, '_set_in_cache') as mock_set_cache:
                # First call should compute and cache
                await ml_service.get_video_recommendations(test_user.id)
                
                # Should attempt to cache the result
                # Note: This depends on implementation details
                assert mock_get_cache.called or mock_set_cache.called
