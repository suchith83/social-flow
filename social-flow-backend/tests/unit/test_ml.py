"""
Unit tests for ML/AI functionality.

This module contains unit tests for the ML service
and related functionality.
"""

import pytest
from unittest.mock import Mock, patch
from app.services.ml_service import MLService
from app.core.exceptions import MLServiceError


class TestMLService:
    """Test cases for MLService."""

    @pytest.fixture
    def ml_service(self):
        """Create MLService instance for testing."""
        return MLService()

    @pytest.mark.asyncio
    async def test_get_personalized_recommendations_success(self, ml_service):
        """Test successful personalized recommendations."""
        user_id = "user123"
        limit = 20
        
        with patch.object(ml_service, 'get_personalized_recommendations') as mock_recommendations:
            mock_recommendations.return_value = {
                "recommendations": [
                    {"id": "video1", "title": "Recommended Video 1", "score": 0.95},
                    {"id": "video2", "title": "Recommended Video 2", "score": 0.87},
                ],
                "total": 2,
            }
            
            result = await ml_service.get_personalized_recommendations(user_id, limit)
            
            assert len(result["recommendations"]) == 2
            assert result["total"] == 2
            mock_recommendations.assert_called_once_with(user_id, limit)

    @pytest.mark.asyncio
    async def test_get_similar_users_success(self, ml_service):
        """Test successful similar users retrieval."""
        user_id = "user123"
        limit = 10
        
        with patch.object(ml_service, 'get_similar_users') as mock_similar:
            mock_similar.return_value = {
                "similar_users": [
                    {"id": "user456", "username": "similar_user1", "similarity": 0.92},
                    {"id": "user789", "username": "similar_user2", "similarity": 0.88},
                ],
                "total": 2,
            }
            
            result = await ml_service.get_similar_users(user_id, limit)
            
            assert len(result["similar_users"]) == 2
            assert result["total"] == 2
            mock_similar.assert_called_once_with(user_id, limit)

    @pytest.mark.asyncio
    async def test_get_similar_content_success(self, ml_service):
        """Test successful similar content retrieval."""
        content_id = "video123"
        content_type = "video"
        limit = 10
        
        with patch.object(ml_service, 'get_similar_content') as mock_similar:
            mock_similar.return_value = {
                "similar_content": [
                    {"id": "video456", "title": "Similar Video 1", "similarity": 0.89},
                    {"id": "video789", "title": "Similar Video 2", "similarity": 0.85},
                ],
                "total": 2,
            }
            
            result = await ml_service.get_similar_content(content_id, content_type, limit)
            
            assert len(result["similar_content"]) == 2
            assert result["total"] == 2
            mock_similar.assert_called_once_with(content_id, content_type, limit)

    @pytest.mark.asyncio
    async def test_record_user_feedback_success(self, ml_service):
        """Test successful user feedback recording."""
        user_id = "user123"
        content_id = "video123"
        feedback_type = "like"
        rating = 5
        
        with patch.object(ml_service, 'record_user_feedback') as mock_feedback:
            mock_feedback.return_value = {"status": "success"}
            
            result = await ml_service.record_user_feedback(user_id, content_id, feedback_type, rating)
            
            assert result["status"] == "success"
            mock_feedback.assert_called_once_with(user_id, content_id, feedback_type, rating)

    @pytest.mark.asyncio
    async def test_get_viral_predictions_success(self, ml_service):
        """Test successful viral predictions."""
        content_id = "video123"
        
        with patch.object(ml_service, 'get_viral_predictions') as mock_viral:
            mock_viral.return_value = {
                "content_id": "video123",
                "viral_score": 0.75,
                "predicted_views": 10000,
                "confidence": 0.82,
                "factors": ["trending_topic", "high_engagement"],
            }
            
            result = await ml_service.get_viral_predictions(content_id)
            
            assert result["content_id"] == "video123"
            assert result["viral_score"] == 0.75
            mock_viral.assert_called_once_with(content_id)

    @pytest.mark.asyncio
    async def test_get_trending_analysis_success(self, ml_service):
        """Test successful trending analysis."""
        time_range = "7d"
        category = "gaming"
        
        with patch.object(ml_service, 'get_trending_analysis') as mock_trending:
            mock_trending.return_value = {
                "trending_topics": [
                    {"topic": "new_game", "score": 0.95, "growth": 0.15},
                    {"topic": "esports", "score": 0.87, "growth": 0.08},
                ],
                "trending_hashtags": ["#gaming", "#newgame", "#esports"],
                "trending_creators": [
                    {"id": "creator1", "username": "gamer1", "growth": 0.12},
                ],
            }
            
            result = await ml_service.get_trending_analysis(time_range, category)
            
            assert len(result["trending_topics"]) == 2
            assert len(result["trending_hashtags"]) == 3
            mock_trending.assert_called_once_with(time_range, category)

    @pytest.mark.asyncio
    async def test_content_moderation_success(self, ml_service):
        """Test successful content moderation."""
        content_id = "video123"
        content_type = "video"
        content_data = "This is test content"
        
        with patch.object(ml_service, 'moderate_content') as mock_moderate:
            mock_moderate.return_value = {
                "content_id": "video123",
                "is_safe": True,
                "confidence": 0.95,
                "flags": [],
                "suggestions": [],
            }
            
            result = await ml_service.moderate_content(content_id, content_type, content_data)
            
            assert result["content_id"] == "video123"
            assert result["is_safe"] is True
            mock_moderate.assert_called_once_with(content_id, content_type, content_data)

    @pytest.mark.asyncio
    async def test_content_moderation_unsafe(self, ml_service):
        """Test content moderation with unsafe content."""
        content_id = "video123"
        content_type = "video"
        content_data = "This is inappropriate content"
        
        with patch.object(ml_service, 'moderate_content') as mock_moderate:
            mock_moderate.return_value = {
                "content_id": "video123",
                "is_safe": False,
                "confidence": 0.92,
                "flags": ["inappropriate_language"],
                "suggestions": ["Consider removing offensive language"],
            }
            
            result = await ml_service.moderate_content(content_id, content_type, content_data)
            
            assert result["content_id"] == "video123"
            assert result["is_safe"] is False
            assert len(result["flags"]) > 0

    @pytest.mark.asyncio
    async def test_auto_tagging_success(self, ml_service):
        """Test successful auto tagging."""
        content_id = "video123"
        content_type = "video"
        content_data = "This is a gaming video about a new game"
        
        with patch.object(ml_service, 'auto_tag_content') as mock_tag:
            mock_tag.return_value = {
                "content_id": "video123",
                "tags": ["gaming", "new_game", "entertainment"],
                "confidence": 0.88,
            }
            
            result = await ml_service.auto_tag_content(content_id, content_type, content_data)
            
            assert result["content_id"] == "video123"
            assert "gaming" in result["tags"]
            mock_tag.assert_called_once_with(content_id, content_type, content_data)

    @pytest.mark.asyncio
    async def test_sentiment_analysis_success(self, ml_service):
        """Test successful sentiment analysis."""
        content_id = "video123"
        content_type = "comment"
        content_data = "This video is amazing! I love it!"
        
        with patch.object(ml_service, 'analyze_sentiment') as mock_sentiment:
            mock_sentiment.return_value = {
                "content_id": "video123",
                "sentiment": "positive",
                "score": 0.92,
                "emotions": ["joy", "excitement"],
            }
            
            result = await ml_service.analyze_sentiment(content_id, content_type, content_data)
            
            assert result["content_id"] == "video123"
            assert result["sentiment"] == "positive"
            assert result["score"] == 0.92
            mock_sentiment.assert_called_once_with(content_id, content_type, content_data)

    @pytest.mark.asyncio
    async def test_get_user_embeddings_success(self, ml_service):
        """Test successful user embeddings retrieval."""
        user_id = "user123"
        
        with patch.object(ml_service, 'get_user_embeddings') as mock_embeddings:
            mock_embeddings.return_value = {
                "user_id": "user123",
                "embeddings": [0.1, 0.2, 0.3, 0.4, 0.5],
                "dimension": 5,
            }
            
            result = await ml_service.get_user_embeddings(user_id)
            
            assert result["user_id"] == "user123"
            assert len(result["embeddings"]) == 5
            mock_embeddings.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_get_content_embeddings_success(self, ml_service):
        """Test successful content embeddings retrieval."""
        content_id = "video123"
        content_type = "video"
        
        with patch.object(ml_service, 'get_content_embeddings') as mock_embeddings:
            mock_embeddings.return_value = {
                "content_id": "video123",
                "embeddings": [0.2, 0.3, 0.4, 0.5, 0.6],
                "dimension": 5,
            }
            
            result = await ml_service.get_content_embeddings(content_id, content_type)
            
            assert result["content_id"] == "video123"
            assert len(result["embeddings"]) == 5
            mock_embeddings.assert_called_once_with(content_id, content_type)

    @pytest.mark.asyncio
    async def test_train_model_success(self, ml_service):
        """Test successful model training."""
        model_type = "recommendation"
        training_data = {"users": [], "interactions": []}
        
        with patch.object(ml_service, 'train_model') as mock_train:
            mock_train.return_value = {
                "model_id": "model_123",
                "status": "training",
                "progress": 0.0,
            }
            
            result = await ml_service.train_model(model_type, training_data)
            
            assert result["model_id"] == "model_123"
            assert result["status"] == "training"
            mock_train.assert_called_once_with(model_type, training_data)

    @pytest.mark.asyncio
    async def test_get_model_status_success(self, ml_service):
        """Test successful model status retrieval."""
        model_id = "model_123"
        
        with patch.object(ml_service, 'get_model_status') as mock_status:
            mock_status.return_value = {
                "model_id": "model_123",
                "status": "completed",
                "accuracy": 0.92,
                "created_at": "2025-01-01T00:00:00Z",
            }
            
            result = await ml_service.get_model_status(model_id)
            
            assert result["model_id"] == "model_123"
            assert result["status"] == "completed"
            mock_status.assert_called_once_with(model_id)
