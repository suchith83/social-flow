"""
ML Pipelines Package - Comprehensive ML pipeline infrastructure.

This package provides end-to-end ML pipeline capabilities including:
- Data preprocessing and cleaning
- Feature engineering and selection
- Model training and optimization
- Model inference and serving
- AI Pipeline Orchestration (Phase 6)
- Batch processing and scheduling
"""

# Original ML pipeline components
from app.ml_pipelines.data_preprocessing import (
    DataCleaner,
    FeatureExtractor,
    DataValidator
)
from app.ml_pipelines.feature_engineering import (
    FeatureTransformer,
    FeatureSelector
)
from app.ml_pipelines.training import (
    ModelTrainer,
    HyperparameterOptimizer
)
from app.ml_pipelines.inference import (
    InferenceEngine,
    ModelServer
)

# Phase 6: AI Pipeline Orchestrator components
from app.ml_pipelines.orchestrator import (
    PipelineOrchestrator,
    PipelineTask,
    PipelineStatus,
    PipelineType,
    get_orchestrator,
)
from app.ml_pipelines.batch_processor import BatchProcessor
from app.ml_pipelines.recommendation_precomputer import RecommendationPrecomputer
from app.ml_pipelines.scheduler import (
    PipelineScheduler,
    ScheduledTask,
    ScheduleFrequency,
    get_scheduler,
)
from app.ml_pipelines.monitor import PipelineMonitor

__all__ = [
    # Data Preprocessing
    "DataCleaner",
    "FeatureExtractor",
    "DataValidator",
    # Feature Engineering
    "FeatureTransformer",
    "FeatureSelector",
    # Training
    "ModelTrainer",
    "HyperparameterOptimizer",
    # Inference
    "InferenceEngine",
    "ModelServer",
    # Phase 6: Orchestration
    "PipelineOrchestrator",
    "PipelineTask",
    "PipelineStatus",
    "PipelineType",
    "get_orchestrator",
    "BatchProcessor",
    "RecommendationPrecomputer",
    "PipelineScheduler",
    "ScheduledTask",
    "ScheduleFrequency",
    "get_scheduler",
    "PipelineMonitor",
]
