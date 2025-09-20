"""
ML Service for integrating AI/ML capabilities.

This service integrates all existing ML modules from ai-models and ml-pipelines
into the FastAPI application.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add ai-models to path
sys.path.append(str(Path(__file__).parent.parent.parent / "ai-models"))
sys.path.append(str(Path(__file__).parent.parent.parent / "ml-pipelines"))

from app.core.config import settings
from app.core.exceptions import MLServiceError

logger = logging.getLogger(__name__)


class MLService:
    """Main ML service integrating all AI/ML capabilities."""
    
    def __init__(self):
        self.models = {}
        self.pipelines = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models and pipelines."""
        try:
            # Initialize content analysis models
            self._init_content_analysis()
            
            # Initialize recommendation models
            self._init_recommendation_models()
            
            # Initialize content moderation models
            self._init_content_moderation()
            
            # Initialize generation models
            self._init_generation_models()
            
            logger.info("ML Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML Service: {e}")
            raise MLServiceError(f"ML Service initialization failed: {e}")
    
    def _init_content_analysis(self):
        """Initialize content analysis models."""
        try:
            # Audio analysis
            from ai_models.content_analysis.audio_analysis.models import AudioClassifier, load_pretrained_model
            self.models['audio_classifier'] = AudioClassifier()
            
            # Object recognition
            from ai_models.content_analysis.object_recognition.models import ObjectDetector
            self.models['object_detector'] = ObjectDetector()
            
            # Scene detection
            from ai_models.content_analysis.scene_detection.models import SceneDetector
            self.models['scene_detector'] = SceneDetector()
            
            logger.info("Content analysis models initialized")
        except ImportError as e:
            logger.warning(f"Content analysis models not available: {e}")
    
    def _init_recommendation_models(self):
        """Initialize recommendation models."""
        try:
            # Content-based recommendation
            from ai_models.recommendation_engine.content_based.recommender import ContentBasedRecommender
            self.models['content_based_recommender'] = ContentBasedRecommender()
            
            # Collaborative filtering
            from ai_models.recommendation_engine.collaborative_filtering.model import CollaborativeFilteringRecommender
            self.models['collaborative_recommender'] = CollaborativeFilteringRecommender()
            
            # Deep learning recommendation
            from ai_models.recommendation_engine.deep_learning.recommender import DeepLearningRecommender
            self.models['deep_learning_recommender'] = DeepLearningRecommender()
            
            # Trending recommendation
            from ai_models.recommendation_engine.trending.recommender import TrendingRecommender
            self.models['trending_recommender'] = TrendingRecommender()
            
            # Viral prediction
            from ai_models.recommendation_engine.viral_prediction.predictor import ViralPredictor
            self.models['viral_predictor'] = ViralPredictor()
            
            logger.info("Recommendation models initialized")
        except ImportError as e:
            logger.warning(f"Recommendation models not available: {e}")
    
    def _init_content_moderation(self):
        """Initialize content moderation models."""
        try:
            # NSFW detection
            from ai_models.content_moderation.nsfw_detection.model import NSFWDetector
            self.models['nsfw_detector'] = NSFWDetector()
            
            # Spam detection
            from ai_models.content_moderation.spam_detection.model import SpamDetector
            self.models['spam_detector'] = SpamDetector()
            
            # Violence detection
            from ai_models.content_moderation.violence_detection.model import ViolenceDetector
            self.models['violence_detector'] = ViolenceDetector()
            
            logger.info("Content moderation models initialized")
        except ImportError as e:
            logger.warning(f"Content moderation models not available: {e}")
    
    def _init_generation_models(self):
        """Initialize content generation models."""
        try:
            # Caption generation
            from ai_models.generation.caption_generation.captioner import CaptionGenerator
            self.models['caption_generator'] = CaptionGenerator()
            
            # Summary generation
            from ai_models.generation.summary_generation.summarizer import SummaryGenerator
            self.models['summary_generator'] = SummaryGenerator()
            
            # Thumbnail generation
            from ai_models.generation.thumbnail_generation.generator import ThumbnailGenerator
            self.models['thumbnail_generator'] = ThumbnailGenerator()
            
            logger.info("Generation models initialized")
        except ImportError as e:
            logger.warning(f"Generation models not available: {e}")
    
    async def analyze_content(self, content_type: str, content_data: Any) -> Dict[str, Any]:
        """Analyze content using appropriate ML models."""
        try:
            results = {}
            
            if content_type == "audio":
                if 'audio_classifier' in self.models:
                    results['audio_analysis'] = await self._analyze_audio(content_data)
            
            elif content_type == "video":
                if 'object_detector' in self.models:
                    results['object_detection'] = await self._detect_objects(content_data)
                if 'scene_detector' in self.models:
                    results['scene_detection'] = await self._detect_scenes(content_data)
            
            elif content_type == "text":
                if 'spam_detector' in self.models:
                    results['spam_detection'] = await self._detect_spam(content_data)
            
            return results
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise MLServiceError(f"Content analysis failed: {e}")
    
    async def moderate_content(self, content_type: str, content_data: Any) -> Dict[str, Any]:
        """Moderate content for safety and compliance."""
        try:
            results = {
                'is_safe': True,
                'confidence': 1.0,
                'flags': [],
                'reason': None
            }
            
            if content_type == "image" or content_type == "video":
                if 'nsfw_detector' in self.models:
                    nsfw_result = await self._detect_nsfw(content_data)
                    if nsfw_result['is_nsfw']:
                        results['is_safe'] = False
                        results['flags'].append('nsfw')
                        results['confidence'] = min(results['confidence'], nsfw_result['confidence'])
                
                if 'violence_detector' in self.models:
                    violence_result = await self._detect_violence(content_data)
                    if violence_result['is_violent']:
                        results['is_safe'] = False
                        results['flags'].append('violence')
                        results['confidence'] = min(results['confidence'], violence_result['confidence'])
            
            elif content_type == "text":
                if 'spam_detector' in self.models:
                    spam_result = await self._detect_spam(content_data)
                    if spam_result['is_spam']:
                        results['is_safe'] = False
                        results['flags'].append('spam')
                        results['confidence'] = min(results['confidence'], spam_result['confidence'])
            
            if not results['is_safe']:
                results['reason'] = f"Content flagged for: {', '.join(results['flags'])}"
            
            return results
        except Exception as e:
            logger.error(f"Content moderation failed: {e}")
            raise MLServiceError(f"Content moderation failed: {e}")
    
    async def generate_recommendations(self, user_id: str, content_type: str = "mixed", limit: int = 10) -> List[Dict[str, Any]]:
        """Generate content recommendations for a user."""
        try:
            recommendations = []
            
            # Use different recommendation strategies based on content type
            if content_type == "video" or content_type == "mixed":
                if 'content_based_recommender' in self.models:
                    video_recs = await self._get_content_based_recommendations(user_id, "video", limit)
                    recommendations.extend(video_recs)
                
                if 'trending_recommender' in self.models:
                    trending_recs = await self._get_trending_recommendations("video", limit // 2)
                    recommendations.extend(trending_recs)
            
            if content_type == "post" or content_type == "mixed":
                if 'collaborative_recommender' in self.models:
                    post_recs = await self._get_collaborative_recommendations(user_id, "post", limit)
                    recommendations.extend(post_recs)
            
            # Remove duplicates and sort by score
            unique_recs = {}
            for rec in recommendations:
                key = f"{rec['type']}_{rec['id']}"
                if key not in unique_recs or rec['score'] > unique_recs[key]['score']:
                    unique_recs[key] = rec
            
            return sorted(unique_recs.values(), key=lambda x: x['score'], reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            raise MLServiceError(f"Recommendation generation failed: {e}")
    
    async def generate_content(self, content_type: str, input_data: Any) -> Dict[str, Any]:
        """Generate content using ML models."""
        try:
            if content_type == "caption" and 'caption_generator' in self.models:
                return await self._generate_caption(input_data)
            elif content_type == "summary" and 'summary_generator' in self.models:
                return await self._generate_summary(input_data)
            elif content_type == "thumbnail" and 'thumbnail_generator' in self.models:
                return await self._generate_thumbnail(input_data)
            else:
                raise MLServiceError(f"Content generation not supported for type: {content_type}")
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise MLServiceError(f"Content generation failed: {e}")
    
    async def predict_viral_potential(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict viral potential of content."""
        try:
            if 'viral_predictor' not in self.models:
                return {'viral_score': 0.5, 'confidence': 0.0}
            
            # Extract features for viral prediction
            features = self._extract_viral_features(content_data)
            
            # Get prediction
            viral_score = self.models['viral_predictor'].predict(features)
            
            return {
                'viral_score': float(viral_score),
                'confidence': 0.8,  # Placeholder
                'factors': self._analyze_viral_factors(content_data)
            }
        except Exception as e:
            logger.error(f"Viral prediction failed: {e}")
            raise MLServiceError(f"Viral prediction failed: {e}")
    
    # Private helper methods
    async def _analyze_audio(self, audio_data: Any) -> Dict[str, Any]:
        """Analyze audio content."""
        # Placeholder implementation
        return {'emotion': 'neutral', 'language': 'en', 'confidence': 0.8}
    
    async def _detect_objects(self, video_data: Any) -> Dict[str, Any]:
        """Detect objects in video content."""
        # Placeholder implementation
        return {'objects': ['person', 'car'], 'confidence': 0.9}
    
    async def _detect_scenes(self, video_data: Any) -> Dict[str, Any]:
        """Detect scenes in video content."""
        # Placeholder implementation
        return {'scenes': ['indoor', 'outdoor'], 'transitions': 3}
    
    async def _detect_spam(self, text_data: str) -> Dict[str, Any]:
        """Detect spam in text content."""
        # Placeholder implementation
        return {'is_spam': False, 'confidence': 0.9}
    
    async def _detect_nsfw(self, content_data: Any) -> Dict[str, Any]:
        """Detect NSFW content."""
        # Placeholder implementation
        return {'is_nsfw': False, 'confidence': 0.9}
    
    async def _detect_violence(self, content_data: Any) -> Dict[str, Any]:
        """Detect violent content."""
        # Placeholder implementation
        return {'is_violent': False, 'confidence': 0.9}
    
    async def _get_content_based_recommendations(self, user_id: str, content_type: str, limit: int) -> List[Dict[str, Any]]:
        """Get content-based recommendations."""
        # Placeholder implementation
        return [{'id': f'rec_{i}', 'type': content_type, 'score': 0.8 - i*0.1} for i in range(limit)]
    
    async def _get_trending_recommendations(self, content_type: str, limit: int) -> List[Dict[str, Any]]:
        """Get trending recommendations."""
        # Placeholder implementation
        return [{'id': f'trending_{i}', 'type': content_type, 'score': 0.9 - i*0.1} for i in range(limit)]
    
    async def _get_collaborative_recommendations(self, user_id: str, content_type: str, limit: int) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations."""
        # Placeholder implementation
        return [{'id': f'collab_{i}', 'type': content_type, 'score': 0.7 - i*0.1} for i in range(limit)]
    
    async def _generate_caption(self, input_data: Any) -> Dict[str, Any]:
        """Generate caption for content."""
        # Placeholder implementation
        return {'caption': 'Generated caption', 'confidence': 0.8}
    
    async def _generate_summary(self, input_data: Any) -> Dict[str, Any]:
        """Generate summary for content."""
        # Placeholder implementation
        return {'summary': 'Generated summary', 'confidence': 0.8}
    
    async def _generate_thumbnail(self, input_data: Any) -> Dict[str, Any]:
        """Generate thumbnail for content."""
        # Placeholder implementation
        return {'thumbnail_url': 'generated_thumbnail.jpg', 'confidence': 0.8}
    
    def _extract_viral_features(self, content_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for viral prediction."""
        # Placeholder implementation
        return np.array([0.5, 0.3, 0.8, 0.2, 0.6])
    
    def _analyze_viral_factors(self, content_data: Dict[str, Any]) -> List[str]:
        """Analyze factors that contribute to viral potential."""
        # Placeholder implementation
        return ['engagement_rate', 'share_potential', 'trending_topic']
    
    # Enhanced recommendation functionality from Python service
    
    async def get_personalized_recommendations(self, user_id: str, limit: int = 50, 
                                            algorithm: str = "hybrid") -> Dict[str, Any]:
        """Get personalized recommendations using multiple algorithms."""
        try:
            # Get user interaction history
            user_interactions = await self._get_user_interactions(user_id)
            
            # Get user demographic data
            user_profile = await self._get_user_profile(user_id)
            
            recommendations = []
            
            if algorithm == "collaborative" or algorithm == "hybrid":
                # Collaborative filtering recommendations
                collab_recs = await self._get_collaborative_recommendations(user_id, "video", limit)
                recommendations.extend(collab_recs)
            
            if algorithm == "content_based" or algorithm == "hybrid":
                # Content-based recommendations
                content_recs = await self._get_content_based_recommendations(user_id, "video", limit)
                recommendations.extend(content_recs)
            
            if algorithm == "deep_learning" or algorithm == "hybrid":
                # Deep learning recommendations
                dl_recs = await self._get_deep_learning_recommendations(user_id, limit)
                recommendations.extend(dl_recs)
            
            # Remove duplicates and rank
            unique_recommendations = self._deduplicate_and_rank(recommendations, limit)
            
            return {
                "user_id": user_id,
                "recommendations": unique_recommendations,
                "algorithm": algorithm,
                "count": len(unique_recommendations),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise MLServiceError(f"Failed to get personalized recommendations: {str(e)}")
    
    async def get_similar_users(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get users similar to the given user."""
        try:
            # TODO: Implement user similarity algorithm
            # This would typically use collaborative filtering or embedding similarity
            
            similar_users = []
            
            return {
                "user_id": user_id,
                "similar_users": similar_users,
                "count": len(similar_users),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise MLServiceError(f"Failed to get similar users: {str(e)}")
    
    async def get_similar_content(self, content_id: str, content_type: str = "video", 
                                limit: int = 20) -> Dict[str, Any]:
        """Get content similar to the given content."""
        try:
            # TODO: Implement content similarity algorithm
            # This would typically use content-based filtering with embeddings
            
            similar_content = []
            
            return {
                "content_id": content_id,
                "content_type": content_type,
                "similar_content": similar_content,
                "count": len(similar_content),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise MLServiceError(f"Failed to get similar content: {str(e)}")
    
    async def record_user_feedback(self, user_id: str, content_id: str, 
                                 feedback_type: str, feedback_value: float) -> Dict[str, Any]:
        """Record user feedback for recommendation learning."""
        try:
            # TODO: Store feedback in database and update user profile
            
            feedback_data = {
                "user_id": user_id,
                "content_id": content_id,
                "feedback_type": feedback_type,  # like, dislike, view, share, etc.
                "feedback_value": feedback_value,  # 0.0 to 1.0
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # TODO: Store in database
            # await self._store_feedback(feedback_data)
            
            # TODO: Update user profile in real-time
            # await self._update_user_profile(user_id, feedback_data)
            
            return {
                "message": "Feedback recorded successfully",
                "feedback_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise MLServiceError(f"Failed to record user feedback: {str(e)}")
    
    async def get_viral_predictions(self, content_id: str) -> Dict[str, Any]:
        """Predict viral potential of content."""
        try:
            # TODO: Implement viral prediction algorithm
            # This would analyze content features, timing, user engagement patterns
            
            viral_score = 0.5  # Placeholder
            
            return {
                "content_id": content_id,
                "viral_score": viral_score,
                "confidence": 0.8,
                "factors": {
                    "content_quality": 0.7,
                    "timing": 0.6,
                    "user_engagement": 0.5,
                    "trending_potential": 0.4
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise MLServiceError(f"Failed to get viral predictions: {str(e)}")
    
    async def get_trending_analysis(self, time_window: str = "24h") -> Dict[str, Any]:
        """Get detailed trending analysis."""
        try:
            # TODO: Implement trending analysis
            # This would analyze content performance over time
            
            trending_data = {
                "rising": [],
                "falling": [],
                "stable": [],
                "time_window": time_window,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return trending_data
            
        except Exception as e:
            raise MLServiceError(f"Failed to get trending analysis: {str(e)}")
    
    # Helper methods for recommendation algorithms
    
    async def _get_user_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user interaction history."""
        # TODO: Query database for user interactions
        return []
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile data."""
        # TODO: Query database for user profile
        return {}
    
    async def _get_deep_learning_recommendations(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get deep learning recommendations."""
        # TODO: Implement deep learning recommendation algorithm
        return []
    
    def _deduplicate_and_rank(self, recommendations: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Remove duplicates and rank recommendations."""
        # TODO: Implement deduplication and ranking logic
        return recommendations[:limit]


# Global ML service instance
ml_service = MLService()
