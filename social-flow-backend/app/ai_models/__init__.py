"""
AI Models Package - Stub Implementations

This package provides stub implementations for AI/ML models with graceful degradation.
When ML features are disabled or models fail to load, these stubs provide fallback functionality.
"""

from typing import List, Dict, Any, Optional
import logging
from uuid import UUID

logger = logging.getLogger(__name__)


class RecommendationModel:
    """Stub recommendation model with fallback to trending content."""
    
    def __init__(self):
        self.loaded = False
        logger.info("RecommendationModel initialized (stub mode)")
    
    async def predict(self, user_id: UUID, limit: int = 20) -> List[UUID]:
        """
        Predict video recommendations for a user.
        
        In stub mode, returns empty list to trigger fallback to trending.
        """
        logger.debug(f"RecommendationModel.predict called for user {user_id} (stub mode)")
        return []
    
    def load(self) -> bool:
        """Load model weights. In stub mode, always returns False."""
        logger.warning("RecommendationModel.load called (stub mode - no real model)")
        self.loaded = False
        return False


class ModerationModel:
    """Stub content moderation model with permissive fallback."""
    
    def __init__(self):
        self.loaded = False
        logger.info("ModerationModel initialized (stub mode)")
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for inappropriate content.
        
        In stub mode, returns safe/clean result to allow content through.
        """
        logger.debug(f"ModerationModel.analyze_text called (stub mode)")
        return {
            "is_safe": True,
            "toxicity_score": 0.0,
            "categories": {
                "toxic": 0.0,
                "severe_toxic": 0.0,
                "obscene": 0.0,
                "threat": 0.0,
                "insult": 0.0,
                "identity_hate": 0.0
            },
            "flagged": False,
            "reason": None
        }
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image for inappropriate content.
        
        In stub mode, returns safe result.
        """
        logger.debug(f"ModerationModel.analyze_image called (stub mode)")
        return {
            "is_safe": True,
            "nsfw_score": 0.0,
            "flagged": False,
            "reason": None
        }
    
    def load(self) -> bool:
        """Load model weights. In stub mode, always returns False."""
        logger.warning("ModerationModel.load called (stub mode - no real model)")
        self.loaded = False
        return False


class AnalysisModel:
    """Stub content analysis model for tags/categories."""
    
    def __init__(self):
        self.loaded = False
        logger.info("AnalysisModel initialized (stub mode)")
    
    async def generate_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """
        Generate tags from text content.
        
        In stub mode, returns empty list to use user-provided tags.
        """
        logger.debug(f"AnalysisModel.generate_tags called (stub mode)")
        return []
    
    async def categorize_content(self, text: str, title: str) -> str:
        """
        Categorize content into predefined categories.
        
        In stub mode, returns 'General' category.
        """
        logger.debug(f"AnalysisModel.categorize_content called (stub mode)")
        return "General"
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        In stub mode, returns empty list.
        """
        logger.debug(f"AnalysisModel.extract_entities called (stub mode)")
        return []
    
    def load(self) -> bool:
        """Load model weights. In stub mode, always returns False."""
        logger.warning("AnalysisModel.load called (stub mode - no real model)")
        self.loaded = False
        return False


class SentimentModel:
    """Stub sentiment analysis model."""
    
    def __init__(self):
        self.loaded = False
        logger.info("SentimentModel initialized (stub mode)")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        In stub mode, returns neutral sentiment.
        """
        logger.debug(f"SentimentModel.analyze_sentiment called (stub mode)")
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "positive": 0.33,
            "neutral": 0.34,
            "negative": 0.33
        }
    
    def load(self) -> bool:
        """Load model weights. In stub mode, always returns False."""
        logger.warning("SentimentModel.load called (stub mode - no real model)")
        self.loaded = False
        return False


# Global model instances (lazily initialized)
_recommendation_model: Optional[RecommendationModel] = None
_moderation_model: Optional[ModerationModel] = None
_analysis_model: Optional[AnalysisModel] = None
_sentiment_model: Optional[SentimentModel] = None


def get_recommendation_model() -> RecommendationModel:
    """Get or create recommendation model instance."""
    global _recommendation_model
    if _recommendation_model is None:
        _recommendation_model = RecommendationModel()
    return _recommendation_model


def get_moderation_model() -> ModerationModel:
    """Get or create moderation model instance."""
    global _moderation_model
    if _moderation_model is None:
        _moderation_model = ModerationModel()
    return _moderation_model


def get_analysis_model() -> AnalysisModel:
    """Get or create analysis model instance."""
    global _analysis_model
    if _analysis_model is None:
        _analysis_model = AnalysisModel()
    return _analysis_model


def get_sentiment_model() -> SentimentModel:
    """Get or create sentiment model instance."""
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = SentimentModel()
    return _sentiment_model


__all__ = [
    "RecommendationModel",
    "ModerationModel",
    "AnalysisModel",
    "SentimentModel",
    "get_recommendation_model",
    "get_moderation_model",
    "get_analysis_model",
    "get_sentiment_model",
]
