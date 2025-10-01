"""
ML Schemas - Pydantic schemas for ML API requests/responses.
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


# ============================================================================
# MODERATION SCHEMAS
# ============================================================================

class ModerationRequest(BaseModel):
    """Request for content moderation."""
    content_id: UUID
    content_type: str = Field(..., description="Type: video, post, comment, image")
    content_data: Dict[str, Any] = Field(..., description="Content data (text, image_url, video_url)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "content_type": "post",
                "content_data": {
                    "text": "This is a post content",
                    "image_url": "https://example.com/image.jpg"
                }
            }
        }


class ModerationResponse(BaseModel):
    """Response from content moderation."""
    content_id: str
    content_type: str
    timestamp: str
    scores: Dict[str, float] = Field(default_factory=dict, description="Moderation scores (nsfw, violence, spam, hate_speech)")
    flags: List[str] = Field(default_factory=list, description="List of flags (nsfw, violence, spam, hate_speech)")
    decision: str = Field(..., description="approved, rejected, flagged, review_required")
    requires_review: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "content_type": "post",
                "timestamp": "2024-01-01T12:00:00",
                "scores": {
                    "nsfw": 0.1,
                    "violence": 0.05,
                    "spam": 0.2,
                    "hate_speech": 0.0
                },
                "flags": [],
                "decision": "approved",
                "requires_review": False
            }
        }


# ============================================================================
# CONTENT ANALYSIS SCHEMAS
# ============================================================================

class ContentAnalysisRequest(BaseModel):
    """Request for content analysis."""
    content_id: UUID
    content_type: str = Field(..., description="Type: video, post, image")
    content_data: Dict[str, Any] = Field(..., description="Content data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "content_type": "video",
                "content_data": {
                    "text": "Amazing video about AI",
                    "video_url": "https://example.com/video.mp4",
                    "thumbnail_url": "https://example.com/thumb.jpg"
                }
            }
        }


class ContentAnalysisResponse(BaseModel):
    """Response from content analysis."""
    content_id: str
    content_type: str
    timestamp: str
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    sentiment: str = Field(default="neutral", description="positive, negative, neutral")
    sentiment_score: float = 0.0
    language: str = "en"
    topics: List[str] = Field(default_factory=list)
    entities: List[Dict[str, str]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "content_type": "video",
                "timestamp": "2024-01-01T12:00:00",
                "tags": ["ai", "machine", "learning", "technology"],
                "categories": ["technology", "education"],
                "sentiment": "positive",
                "sentiment_score": 0.8,
                "language": "en",
                "topics": ["artificial intelligence", "neural networks"],
                "entities": [
                    {"text": "OpenAI", "type": "organization"},
                    {"text": "GPT-4", "type": "product"}
                ]
            }
        }


# ============================================================================
# RECOMMENDATION SCHEMAS
# ============================================================================

class RecommendationRequest(BaseModel):
    """Request for content recommendations."""
    user_id: Optional[UUID] = None
    content_type: str = Field(default="video", description="Type: video, post, user")
    limit: int = Field(default=20, ge=1, le=50)
    exclude_ids: List[UUID] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "content_type": "video",
                "limit": 20,
                "exclude_ids": []
            }
        }


class RecommendationResponse(BaseModel):
    """Response with content recommendations."""
    user_id: UUID
    content_type: str
    recommendations: List[str] = Field(..., description="List of content IDs")
    count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "content_type": "video",
                "recommendations": [
                    "223e4567-e89b-12d3-a456-426614174000",
                    "323e4567-e89b-12d3-a456-426614174000"
                ],
                "count": 2
            }
        }


# ============================================================================
# TRENDING & VIRAL SCHEMAS
# ============================================================================

class TrendingResponse(BaseModel):
    """Response with trending content."""
    content_type: str
    time_window: str
    trending: List[str] = Field(..., description="List of trending content IDs")
    count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_type": "video",
                "time_window": "24h",
                "trending": [
                    "223e4567-e89b-12d3-a456-426614174000",
                    "323e4567-e89b-12d3-a456-426614174000"
                ],
                "count": 2
            }
        }


class ViralPredictionRequest(BaseModel):
    """Request for viral prediction."""
    content_id: UUID
    content_data: Dict[str, Any] = Field(..., description="Content metadata")
    engagement_data: Optional[Dict[str, Any]] = Field(default=None, description="Early engagement metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "content_data": {
                    "title": "Amazing video",
                    "duration": 120,
                    "created_at": "2024-01-01T12:00:00"
                },
                "engagement_data": {
                    "views": 1000,
                    "likes": 100,
                    "comments": 20,
                    "shares": 50
                }
            }
        }


class ViralPredictionResponse(BaseModel):
    """Response from viral prediction."""
    content_id: str
    timestamp: str
    viral_score: float = Field(..., ge=0.0, le=1.0, description="Viral probability (0-1)")
    is_likely_viral: bool
    factors: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-01T12:00:00",
                "viral_score": 0.85,
                "is_likely_viral": True,
                "factors": {
                    "engagement_rate": 0.1,
                    "share_velocity": 0.05,
                    "timing_score": 0.8
                },
                "confidence": 0.75
            }
        }


# ============================================================================
# MODEL MANAGEMENT SCHEMAS
# ============================================================================

class ModelInfoResponse(BaseModel):
    """Response with ML model information."""
    loaded_models: List[str]
    model_count: int
    cache_ttl: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "loaded_models": ["nsfw_detector", "spam_detector"],
                "model_count": 2,
                "cache_ttl": 3600
            }
        }


class ModelLoadRequest(BaseModel):
    """Request to load a model."""
    model_name: str = Field(..., description="Name of model to load")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "nsfw_detector"
            }
        }


class ModelResponse(BaseModel):
    """Response for model operations."""
    message: str
    model_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Model loaded successfully",
                "model_name": "nsfw_detector"
            }
        }
