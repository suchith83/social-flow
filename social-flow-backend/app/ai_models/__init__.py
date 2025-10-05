"""
AI Models Package - Advanced AI/ML capabilities for Social Flow.

This package provides comprehensive AI/ML functionality including:
- Content moderation and safety
- Intelligent recommendations
- Video analysis and processing  
- Sentiment and emotion analysis
- Trending prediction and forecasting
"""

from app.ai_models.content_moderation import (
    NSFWDetector,
    SpamDetector,
    ViolenceDetector,
    ToxicityDetector
)
from app.ai_models.recommendation import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    DeepLearningRecommender,
    TrendingRecommender,
    ViralPredictor,
    HybridRecommender
)
from app.ai_models.video_analysis import (
    SceneDetector,
    ObjectDetector,
    ActionRecognizer,
    VideoQualityAnalyzer,
    ThumbnailGenerator
)
from app.ai_models.sentiment_analysis import (
    SentimentAnalyzer,
    EmotionDetector,
    IntentRecognizer
)
from app.ai_models.trending_prediction import (
    TrendPredictor,
    TrendAnalyzer,
    EngagementForecaster
)

__all__ = [
    # Content Moderation
    "NSFWDetector",
    "SpamDetector",
    "ViolenceDetector",
    "ToxicityDetector",
    # Recommendations
    "ContentBasedRecommender",
    "CollaborativeFilteringRecommender",
    "DeepLearningRecommender",
    "TrendingRecommender",
    "ViralPredictor",
    "HybridRecommender",
    # Video Analysis
    "SceneDetector",
    "ObjectDetector",
    "ActionRecognizer",
    "VideoQualityAnalyzer",
    "ThumbnailGenerator",
    # Sentiment Analysis
    "SentimentAnalyzer",
    "EmotionDetector",
    "IntentRecognizer",
    # Trending Prediction
    "TrendPredictor",
    "TrendAnalyzer",
    "EngagementForecaster"
]
