"""TypedDict definitions for MLService standardized outputs.

References AI_ML_ARCHITECTURE.md Schemas section.
"""
from __future__ import annotations
from typing import TypedDict, List, Dict, Any

class RecommendationItem(TypedDict, total=False):
    id: str
    type: str
    score: float
    algorithm: str

class ViralPredictionFactors(TypedDict, total=False):
    engagement_velocity: float
    creator_influence: float
    content_richness: float

class ViralPrediction(TypedDict, total=False):
    viral_score: float
    factors: ViralPredictionFactors

class ModerationScores(TypedDict, total=False):
    nsfw: float
    violence: float
    toxicity: float
    spam: float

class ModerationAggregate(TypedDict, total=False):
    scores: ModerationScores
    risk_score: float
    flags: List[str]
    is_safe: bool

class StandardResponse(TypedDict, total=False):
    success: bool
    data: Any
    meta: Dict[str, Any]
