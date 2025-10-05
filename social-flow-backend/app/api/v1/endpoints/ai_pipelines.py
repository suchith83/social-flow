"""
AI and ML Pipeline API endpoints.

Provides access to:
- Advanced video recommendations (transformer, neural CF, graph, smart)
- Pipeline orchestrator (task submission, monitoring, scheduling)
- Batch processing (video analysis, recommendation pre-computation)
- Pipeline health and metrics
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_current_user_optional,
    get_db,
    require_admin,
)
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.services.recommendation_service import RecommendationService
from app.ml_pipelines.orchestrator import get_orchestrator, PipelineType
from app.ml_pipelines.scheduler import get_scheduler
from app.ml_pipelines.monitor import PipelineMonitor

router = APIRouter()


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class RecommendationAlgorithm(str):
    """Available recommendation algorithms."""
    HYBRID = "hybrid"
    TRENDING = "trending"
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    TRANSFORMER = "transformer"  # Advanced: BERT-based
    NEURAL_CF = "neural_cf"  # Advanced: Neural Collaborative Filtering
    GRAPH = "graph"  # Advanced: Graph Neural Network
    SMART = "smart"  # Advanced: Multi-armed Bandit


class VideoRecommendationRequest(BaseModel):
    """Request for video recommendations."""
    algorithm: str = Field(
        default="hybrid",
        description="Recommendation algorithm to use",
        examples=["hybrid", "transformer", "smart"]
    )
    limit: int = Field(default=20, ge=1, le=100, description="Number of recommendations")
    exclude_video_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Video IDs to exclude from recommendations"
    )


class VideoRecommendationResponse(BaseModel):
    """Response for video recommendations."""
    recommendations: List[Dict[str, Any]]
    algorithm: str
    count: int
    cached: bool
    ml_available: bool


class PipelineTaskSubmit(BaseModel):
    """Request to submit a pipeline task."""
    pipeline_type: str = Field(
        description="Type of pipeline task",
        examples=["batch_video_analysis", "batch_recommendation_precompute"]
    )
    name: str = Field(description="Human-readable task name")
    config: Dict[str, Any] = Field(description="Task configuration parameters")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1=highest)")


class PipelineTaskResponse(BaseModel):
    """Response for pipeline task submission."""
    task_id: str
    status: str
    message: str


class PipelineTaskStatus(BaseModel):
    """Pipeline task status response."""
    task_id: str
    pipeline_type: str
    name: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    result: Optional[Dict[str, Any]]


class PipelineQueueStatus(BaseModel):
    """Pipeline queue status response."""
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    max_concurrent: int
    is_running: bool


class PipelineHealthResponse(BaseModel):
    """Pipeline health status response."""
    status: str
    error_rate: float
    total_tasks_24h: int
    total_errors_24h: int
    timestamp: str


class BatchVideoAnalysisRequest(BaseModel):
    """Request for batch video analysis."""
    video_ids: List[UUID] = Field(description="List of video IDs to analyze")
    force_reanalysis: bool = Field(
        default=False,
        description="Force re-analysis even if results exist"
    )
    priority: int = Field(default=5, ge=1, le=10, description="Task priority")


class BatchRecommendationRequest(BaseModel):
    """Request for batch recommendation pre-computation."""
    user_ids: Optional[List[UUID]] = Field(
        default=None,
        description="User IDs to pre-compute for (None = active users)"
    )
    algorithms: List[str] = Field(
        default=["hybrid", "smart"],
        description="Algorithms to use"
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Max users to process"
    )
    priority: int = Field(default=5, ge=1, le=10, description="Task priority")


# ============================================================================
# Video Recommendation Endpoints
# ============================================================================

@router.get("/recommendations", response_model=VideoRecommendationResponse)
async def get_video_recommendations(
    *,
    db: AsyncSession = Depends(get_db),
    algorithm: str = Query(
        default="hybrid",
        description="Recommendation algorithm",
        pattern="^(hybrid|trending|collaborative|content_based|collaborative_filtering|trending_weighted|personalized_trending|transformer|neural_cf|graph|smart)$"
    ),
    limit: int = Query(default=20, ge=1, le=100),
    exclude_ids: Optional[List[UUID]] = Query(default=None),
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Get personalized video recommendations.
    
    **Algorithms:**
    
    **Traditional:**
    - `hybrid`: Multi-signal combination (40% collaborative, 30% content, 20% trending, 10% diversity)
    - `trending`: Popular content based on recent engagement
    - `collaborative`: Based on similar users' preferences
    - `content_based`: Based on video attributes and user history
    
    **Advanced (ML-powered):**
    - `transformer`: BERT-based semantic matching for deep content understanding
    - `neural_cf`: Neural collaborative filtering with deep learning
    - `graph`: Graph neural network leveraging social connections
    - `smart`: Multi-armed bandit for adaptive algorithm selection
    
    **Enhanced Hybrid (with ML):**
    - When ML available: 20% transformer + 20% neural_cf + 20% graph + 20% collaborative + 10% trending + 10% diversity
    - Automatic fallback to traditional mix when ML unavailable
    
    Returns personalized recommendations optimized for user engagement.
    """
    rec_service = RecommendationService(db)
    
    try:
        # Get recommendations
        result = await rec_service.get_video_recommendations(
            user_id=current_user.id if current_user else None,
            limit=limit,
            algorithm=algorithm,
            exclude_ids=exclude_ids,
        )
        
        # Check if ML is available
        from app.services.recommendation_service import ADVANCED_ML_AVAILABLE
        
        return VideoRecommendationResponse(
            recommendations=result.get("recommendations", []),
            algorithm=algorithm,
            count=len(result.get("recommendations", [])),
            cached=result.get("cached", False),
            ml_available=ADVANCED_ML_AVAILABLE,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@router.get("/recommendations/algorithms", response_model=Dict[str, Any])
async def list_recommendation_algorithms(
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    List available recommendation algorithms with descriptions.
    
    Returns information about which algorithms are available,
    including whether advanced ML algorithms are enabled.
    """
    from app.services.recommendation_service import ADVANCED_ML_AVAILABLE
    
    algorithms = {
        "traditional": [
            {
                "name": "hybrid",
                "display_name": "Hybrid Recommendations",
                "description": "Combines multiple signals for balanced recommendations",
                "available": True,
                "performance": "Best overall quality",
            },
            {
                "name": "trending",
                "display_name": "Trending",
                "description": "Popular content based on recent engagement",
                "available": True,
                "performance": "Fast, good for discovery",
            },
            {
                "name": "collaborative",
                "display_name": "Collaborative Filtering",
                "description": "Based on similar users' preferences",
                "available": True,
                "performance": "Good for established users",
            },
            {
                "name": "content_based",
                "display_name": "Content-Based",
                "description": "Based on video attributes and tags",
                "available": True,
                "performance": "Good for niche content",
            },
        ],
        "advanced": [
            {
                "name": "transformer",
                "display_name": "Transformer (BERT)",
                "description": "Deep semantic understanding with BERT embeddings",
                "available": ADVANCED_ML_AVAILABLE,
                "performance": "Excellent semantic matching",
                "requires": "ML service",
            },
            {
                "name": "neural_cf",
                "display_name": "Neural Collaborative Filtering",
                "description": "Deep neural networks for complex user-item interactions",
                "available": ADVANCED_ML_AVAILABLE,
                "performance": "Best personalization",
                "requires": "ML service",
            },
            {
                "name": "graph",
                "display_name": "Graph Neural Network",
                "description": "Social network-aware recommendations",
                "available": ADVANCED_ML_AVAILABLE,
                "performance": "Best for viral content",
                "requires": "ML service + social graph",
            },
            {
                "name": "smart",
                "display_name": "Smart (Multi-Armed Bandit)",
                "description": "Adaptive algorithm selection based on performance",
                "available": ADVANCED_ML_AVAILABLE,
                "performance": "Continuously optimizing",
                "requires": "ML service",
            },
        ],
        "ml_service_status": {
            "available": ADVANCED_ML_AVAILABLE,
            "message": "Advanced ML algorithms enabled" if ADVANCED_ML_AVAILABLE
                      else "Advanced ML algorithms unavailable - using traditional algorithms"
        }
    }
    
    return algorithms


# ============================================================================
# Pipeline Orchestrator Endpoints (Admin Only)
# ============================================================================

@router.post("/pipelines/tasks", response_model=PipelineTaskResponse)
async def submit_pipeline_task(
    *,
    task_data: PipelineTaskSubmit,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Submit a new pipeline task (Admin only).
    
    **Pipeline Types:**
    - `batch_video_analysis`: Analyze multiple videos with AI
    - `batch_recommendation_precompute`: Pre-compute recommendations for users
    - `cache_warming`: Warm recommendation cache
    - `data_cleanup`: Clean old cache and expired data
    
    **Priority:** 1 (highest) to 10 (lowest)
    
    Returns task ID for tracking.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # Validate pipeline type
        valid_types = [
            "batch_video_analysis",
            "batch_recommendation_precompute",
            "cache_warming",
            "data_cleanup",
        ]
        
        if task_data.pipeline_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid pipeline type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Convert to PipelineType enum
        pipeline_type = PipelineType(task_data.pipeline_type)
        
        # Submit task
        task_id = await orchestrator.submit_task(
            pipeline_type=pipeline_type,
            name=task_data.name,
            config=task_data.config,
            priority=task_data.priority,
        )
        
        return PipelineTaskResponse(
            task_id=task_id,
            status="pending",
            message=f"Task '{task_data.name}' submitted successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )


@router.get("/pipelines/tasks/{task_id}", response_model=PipelineTaskStatus)
async def get_pipeline_task_status(
    *,
    task_id: str,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Get status of a pipeline task (Admin only).
    
    Returns detailed task information including:
    - Current status (pending/running/completed/failed/cancelled)
    - Progress percentage
    - Execution timestamps
    - Results or error details
    """
    try:
        orchestrator = await get_orchestrator()
        task_status = await orchestrator.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return PipelineTaskStatus(**task_status)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.delete("/pipelines/tasks/{task_id}", response_model=SuccessResponse)
async def cancel_pipeline_task(
    *,
    task_id: str,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Cancel a pipeline task (Admin only).
    
    Cancels a pending or running task.
    Returns success if task was cancelled.
    """
    try:
        orchestrator = await get_orchestrator()
        cancelled = await orchestrator.cancel_task(task_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found or already completed"
            )
        
        return SuccessResponse(
            success=True,
            message=f"Task {task_id} cancelled successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@router.get("/pipelines/queue", response_model=PipelineQueueStatus)
async def get_pipeline_queue_status(
    admin: User = Depends(require_admin),
) -> Any:
    """
    Get pipeline queue status (Admin only).
    
    Returns:
    - Number of pending, running, and completed tasks
    - Maximum concurrent tasks
    - Whether orchestrator is running
    """
    try:
        orchestrator = await get_orchestrator()
        queue_status = await orchestrator.get_queue_status()
        
        return PipelineQueueStatus(**queue_status)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )


@router.get("/pipelines/health", response_model=PipelineHealthResponse)
async def get_pipeline_health(
    admin: User = Depends(require_admin),
) -> Any:
    """
    Get pipeline health status (Admin only).
    
    Returns:
    - Overall health status (healthy/warning/degraded/critical)
    - Error rate (last 24 hours)
    - Task execution counts
    - Timestamp
    """
    try:
        monitor = PipelineMonitor()
        health = await monitor.get_health_status()
        
        return PipelineHealthResponse(**health)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}"
        )


@router.get("/pipelines/metrics", response_model=Dict[str, Any])
async def get_pipeline_metrics(
    *,
    pipeline_type: Optional[str] = Query(None, description="Filter by pipeline type"),
    days: int = Query(default=7, ge=1, le=30, description="Number of days"),
    admin: User = Depends(require_admin),
) -> Any:
    """
    Get pipeline metrics (Admin only).
    
    Returns detailed metrics including:
    - Task execution counts by type and status
    - Duration statistics (avg, min, max)
    - Daily breakdown
    - Error rates
    """
    try:
        monitor = PipelineMonitor()
        metrics = await monitor.get_metrics(pipeline_type=pipeline_type, days=days)
        
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/pipelines/performance", response_model=Dict[str, Any])
async def get_pipeline_performance_report(
    admin: User = Depends(require_admin),
) -> Any:
    """
    Get 7-day pipeline performance report (Admin only).
    
    Returns:
    - Total tasks executed
    - Success/error rates
    - Average tasks per day
    - Performance trends
    """
    try:
        monitor = PipelineMonitor()
        report = await monitor.get_performance_report()
        
        return report
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance report: {str(e)}"
        )


# ============================================================================
# Batch Processing Convenience Endpoints (Admin Only)
# ============================================================================

@router.post("/pipelines/batch/videos", response_model=PipelineTaskResponse)
async def batch_analyze_videos(
    *,
    request: BatchVideoAnalysisRequest,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Batch analyze videos with AI (Admin only).
    
    Convenience endpoint for submitting video analysis tasks.
    Analyzes multiple videos with YOLO, Whisper, CLIP, and Scene Detection.
    
    - **video_ids**: List of video IDs to analyze
    - **force_reanalysis**: Re-analyze even if results exist
    - **priority**: Task priority (1=highest, 10=lowest)
    
    Returns task ID for tracking progress.
    """
    try:
        orchestrator = await get_orchestrator()
        
        task_id = await orchestrator.submit_task(
            pipeline_type=PipelineType.BATCH_VIDEO_ANALYSIS,
            name=f"Batch Video Analysis ({len(request.video_ids)} videos)",
            config={
                "video_ids": [str(vid) for vid in request.video_ids],
                "force_reanalysis": request.force_reanalysis,
            },
            priority=request.priority,
        )
        
        return PipelineTaskResponse(
            task_id=task_id,
            status="pending",
            message=f"Batch video analysis submitted for {len(request.video_ids)} videos"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit batch analysis: {str(e)}"
        )


@router.post("/pipelines/batch/recommendations", response_model=PipelineTaskResponse)
async def batch_precompute_recommendations(
    *,
    request: BatchRecommendationRequest,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Batch pre-compute recommendations (Admin only).
    
    Convenience endpoint for pre-computing recommendations.
    Pre-computes and caches recommendations for multiple users.
    
    - **user_ids**: User IDs (None = active users)
    - **algorithms**: Algorithms to use
    - **limit**: Max users to process
    - **priority**: Task priority
    
    Returns task ID for tracking progress.
    """
    try:
        orchestrator = await get_orchestrator()
        
        # If no user_ids provided, will use active users (handled by precomputer)
        user_ids_str = [str(uid) for uid in request.user_ids] if request.user_ids else []
        
        task_id = await orchestrator.submit_task(
            pipeline_type=PipelineType.BATCH_RECOMMENDATION_PRECOMPUTE,
            name=f"Batch Recommendation Pre-compute ({request.limit} users)",
            config={
                "user_ids": user_ids_str,
                "algorithms": request.algorithms,
                "limit": request.limit,
            },
            priority=request.priority,
        )
        
        return PipelineTaskResponse(
            task_id=task_id,
            status="pending",
            message=f"Batch recommendation pre-computation submitted for {request.limit} users"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit batch pre-computation: {str(e)}"
        )


@router.post("/pipelines/cache/warm", response_model=PipelineTaskResponse)
async def warm_recommendation_cache(
    *,
    limit: int = Query(default=1000, ge=1, le=10000),
    algorithm: str = Query(default="hybrid"),
    priority: int = Query(default=5, ge=1, le=10),
    admin: User = Depends(require_admin),
) -> Any:
    """
    Warm recommendation cache (Admin only).
    
    Pre-computes recommendations for most active users
    to improve response times during peak hours.
    
    - **limit**: Number of users to warm cache for
    - **algorithm**: Algorithm to use
    - **priority**: Task priority
    """
    try:
        orchestrator = await get_orchestrator()
        
        task_id = await orchestrator.submit_task(
            pipeline_type=PipelineType.CACHE_WARMING,
            name=f"Cache Warming ({limit} users, {algorithm})",
            config={
                "cache_types": ["recommendations"],
                "limit": limit,
                "algorithm": algorithm,
            },
            priority=priority,
        )
        
        return PipelineTaskResponse(
            task_id=task_id,
            status="pending",
            message=f"Cache warming submitted for {limit} users"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit cache warming: {str(e)}"
        )


# ============================================================================
# Pipeline Scheduler Endpoints (Admin Only)
# ============================================================================

@router.get("/pipelines/schedule", response_model=Dict[str, Any])
async def get_pipeline_schedule(
    admin: User = Depends(require_admin),
) -> Any:
    """
    Get pipeline schedule status (Admin only).
    
    Returns information about all scheduled tasks including:
    - Task names and types
    - Schedule frequency
    - Last run and next run times
    - Enabled/disabled status
    """
    try:
        scheduler = get_scheduler()
        schedule_status = scheduler.get_schedule_status()
        
        return schedule_status
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get schedule status: {str(e)}"
        )


@router.post("/pipelines/schedule/{task_name}/enable", response_model=SuccessResponse)
async def enable_scheduled_task(
    *,
    task_name: str,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Enable a scheduled task (Admin only).
    
    Re-enables a previously disabled scheduled task.
    """
    try:
        scheduler = get_scheduler()
        enabled = scheduler.enable_task(task_name)
        
        if not enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled task '{task_name}' not found"
            )
        
        return SuccessResponse(
            success=True,
            message=f"Scheduled task '{task_name}' enabled"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable task: {str(e)}"
        )


@router.post("/pipelines/schedule/{task_name}/disable", response_model=SuccessResponse)
async def disable_scheduled_task(
    *,
    task_name: str,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Disable a scheduled task (Admin only).
    
    Disables a scheduled task without removing it.
    Task will not run until re-enabled.
    """
    try:
        scheduler = get_scheduler()
        disabled = scheduler.disable_task(task_name)
        
        if not disabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scheduled task '{task_name}' not found"
            )
        
        return SuccessResponse(
            success=True,
            message=f"Scheduled task '{task_name}' disabled"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable task: {str(e)}"
        )
