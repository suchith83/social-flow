"""
AI Pipeline Orchestrator.

Central coordinator for all AI/ML pipelines, managing task execution,
resource allocation, and pipeline dependencies.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from enum import Enum
import json

from app.ml_pipelines.batch_processor import BatchProcessor
from app.ml_pipelines.recommendation_precomputer import RecommendationPrecomputer
from app.ml_pipelines.monitor import PipelineMonitor
from app.core.redis import get_redis

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineType(str, Enum):
    """Types of AI pipelines."""
    BATCH_VIDEO_ANALYSIS = "batch_video_analysis"
    BATCH_RECOMMENDATION_PRECOMPUTE = "batch_recommendation_precompute"
    MODEL_TRAINING = "model_training"
    MODEL_UPDATE = "model_update"
    CACHE_WARMING = "cache_warming"
    DATA_CLEANUP = "data_cleanup"


class PipelineTask:
    """Represents a single pipeline task."""
    
    def __init__(
        self,
        task_id: str,
        pipeline_type: PipelineType,
        name: str,
        config: Dict[str, Any],
        priority: int = 5,
    ):
        self.task_id = task_id
        self.pipeline_type = pipeline_type
        self.name = name
        self.config = config
        self.priority = priority
        self.status = PipelineStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.progress: float = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "pipeline_type": self.pipeline_type.value,
            "name": self.name,
            "config": self.config,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result": self.result,
            "progress": self.progress,
        }


class PipelineOrchestrator:
    """
    Central orchestrator for AI/ML pipelines.
    
    Manages execution of various AI pipeline tasks including:
    - Batch video analysis
    - Recommendation pre-computation
    - Model training and updates
    - Cache warming
    - Data cleanup
    
    Features:
    - Priority-based task queue
    - Concurrent execution with resource limits
    - Progress tracking and monitoring
    - Error handling and retry logic
    - Task cancellation support
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        redis_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent pipeline tasks
            redis_ttl: TTL for task status in Redis (seconds)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.redis_ttl = redis_ttl
        
        # Task queue and tracking
        self.pending_tasks: List[PipelineTask] = []
        self.running_tasks: Dict[str, PipelineTask] = {}
        self.completed_tasks: Dict[str, PipelineTask] = {}
        
        # Pipeline components
        self.batch_processor: Optional[BatchProcessor] = None
        self.recommendation_precomputer: Optional[RecommendationPrecomputer] = None
        self.monitor = PipelineMonitor()
        
        # Execution control
        self._shutdown = False
        self._executor_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"Pipeline orchestrator initialized with max {max_concurrent_tasks} concurrent tasks"
        )
    
    async def initialize(self) -> None:
        """Initialize pipeline components."""
        try:
            self.batch_processor = BatchProcessor()
            self.recommendation_precomputer = RecommendationPrecomputer()
            
            # Start executor loop
            self._executor_task = asyncio.create_task(self._executor_loop())
            
            logger.info("Pipeline orchestrator components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline orchestrator: {e}", exc_info=True)
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown orchestrator."""
        logger.info("Shutting down pipeline orchestrator")
        self._shutdown = True
        
        # Cancel running tasks
        for task_id, task in self.running_tasks.items():
            logger.info(f"Cancelling running task: {task_id}")
            task.status = PipelineStatus.CANCELLED
        
        # Wait for executor to finish
        if self._executor_task:
            await self._executor_task
        
        logger.info("Pipeline orchestrator shutdown complete")
    
    async def submit_task(
        self,
        pipeline_type: PipelineType,
        name: str,
        config: Dict[str, Any],
        priority: int = 5,
    ) -> str:
        """
        Submit a new pipeline task.
        
        Args:
            pipeline_type: Type of pipeline to execute
            name: Human-readable task name
            config: Task configuration parameters
            priority: Task priority (1=highest, 10=lowest)
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid4())
        task = PipelineTask(task_id, pipeline_type, name, config, priority)
        
        # Add to pending queue (sorted by priority)
        self.pending_tasks.append(task)
        self.pending_tasks.sort(key=lambda t: t.priority)
        
        # Store in Redis for persistence
        await self._save_task_to_redis(task)
        
        logger.info(
            f"Task submitted: {task_id} ({pipeline_type.value}) "
            f"priority={priority}, queue_size={len(self.pending_tasks)}"
        )
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        # Check running tasks
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].to_dict()
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        # Check pending tasks
        for task in self.pending_tasks:
            if task.task_id == task_id:
                return task.to_dict()
        
        # Check Redis
        redis = await get_redis()
        if redis:
            task_data = await redis.get(f"pipeline:task:{task_id}")
            if task_data:
                return json.loads(task_data)
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        # Check pending tasks
        for i, task in enumerate(self.pending_tasks):
            if task.task_id == task_id:
                task.status = PipelineStatus.CANCELLED
                self.pending_tasks.pop(i)
                await self._save_task_to_redis(task)
                logger.info(f"Cancelled pending task: {task_id}")
                return True
        
        # Check running tasks (mark for cancellation)
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = PipelineStatus.CANCELLED
            await self._save_task_to_redis(task)
            logger.info(f"Marked running task for cancellation: {task_id}")
            return True
        
        return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "is_running": not self._shutdown,
            "pending_task_ids": [t.task_id for t in self.pending_tasks],
            "running_task_ids": list(self.running_tasks.keys()),
        }
    
    async def _executor_loop(self) -> None:
        """Main executor loop for processing tasks."""
        logger.info("Pipeline executor loop started")
        
        while not self._shutdown:
            try:
                # Check if we can start new tasks
                if len(self.running_tasks) < self.max_concurrent_tasks and self.pending_tasks:
                    # Get highest priority task
                    task = self.pending_tasks.pop(0)
                    
                    # Start task execution
                    asyncio.create_task(self._execute_task(task))
                
                # Sleep briefly before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in executor loop: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("Pipeline executor loop stopped")
    
    async def _execute_task(self, task: PipelineTask) -> None:
        """Execute a pipeline task."""
        task.status = PipelineStatus.RUNNING
        task.started_at = datetime.utcnow()
        self.running_tasks[task.task_id] = task
        
        logger.info(f"Starting task execution: {task.task_id} ({task.pipeline_type.value})")
        
        try:
            # Execute based on pipeline type
            if task.pipeline_type == PipelineType.BATCH_VIDEO_ANALYSIS:
                result = await self._execute_batch_video_analysis(task)
            elif task.pipeline_type == PipelineType.BATCH_RECOMMENDATION_PRECOMPUTE:
                result = await self._execute_batch_recommendation_precompute(task)
            elif task.pipeline_type == PipelineType.CACHE_WARMING:
                result = await self._execute_cache_warming(task)
            elif task.pipeline_type == PipelineType.DATA_CLEANUP:
                result = await self._execute_data_cleanup(task)
            else:
                raise ValueError(f"Unknown pipeline type: {task.pipeline_type}")
            
            # Task completed successfully
            task.status = PipelineStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            task.progress = 100.0
            
            logger.info(
                f"Task completed: {task.task_id} in "
                f"{(task.completed_at - task.started_at).total_seconds():.2f}s"
            )
            
        except Exception as e:
            # Task failed
            task.status = PipelineStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error = str(e)
            
            logger.error(f"Task failed: {task.task_id} - {e}", exc_info=True)
        
        finally:
            # Move from running to completed
            self.running_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task
            
            # Save to Redis
            await self._save_task_to_redis(task)
            
            # Update monitoring
            await self.monitor.record_task_execution(task)
    
    async def _execute_batch_video_analysis(self, task: PipelineTask) -> Dict[str, Any]:
        """Execute batch video analysis pipeline."""
        if not self.batch_processor:
            raise RuntimeError("Batch processor not initialized")
        
        video_ids = task.config.get("video_ids", [])
        force_reanalysis = task.config.get("force_reanalysis", False)
        
        result = await self.batch_processor.process_videos(
            video_ids=video_ids,
            force_reanalysis=force_reanalysis,
            progress_callback=lambda p: self._update_task_progress(task.task_id, p),
        )
        
        return result
    
    async def _execute_batch_recommendation_precompute(self, task: PipelineTask) -> Dict[str, Any]:
        """Execute batch recommendation pre-computation pipeline."""
        if not self.recommendation_precomputer:
            raise RuntimeError("Recommendation precomputer not initialized")
        
        user_ids = task.config.get("user_ids", [])
        algorithms = task.config.get("algorithms", ["hybrid"])
        
        result = await self.recommendation_precomputer.precompute_recommendations(
            user_ids=user_ids,
            algorithms=algorithms,
            progress_callback=lambda p: self._update_task_progress(task.task_id, p),
        )
        
        return result
    
    async def _execute_cache_warming(self, task: PipelineTask) -> Dict[str, Any]:
        """Execute cache warming pipeline."""
        cache_types = task.config.get("cache_types", ["recommendations"])
        
        warmed_count = 0
        for cache_type in cache_types:
            if cache_type == "recommendations":
                # Warm recommendation cache
                result = await self.recommendation_precomputer.warm_cache(
                    limit=task.config.get("limit", 1000)
                )
                warmed_count += result.get("precomputed_count", 0)
        
        return {
            "cache_types": cache_types,
            "warmed_count": warmed_count,
        }
    
    async def _execute_data_cleanup(self, task: PipelineTask) -> Dict[str, Any]:
        """Execute data cleanup pipeline."""
        cleanup_types = task.config.get("cleanup_types", ["old_cache"])
        
        cleaned_count = 0
        # Implementation would clean up old cache entries, expired tasks, etc.
        
        return {
            "cleanup_types": cleanup_types,
            "cleaned_count": cleaned_count,
        }
    
    def _update_task_progress(self, task_id: str, progress: float) -> None:
        """Update task progress."""
        if task_id in self.running_tasks:
            self.running_tasks[task_id].progress = progress
    
    async def _save_task_to_redis(self, task: PipelineTask) -> None:
        """Save task status to Redis."""
        try:
            redis = await get_redis()
            if redis:
                await redis.setex(
                    f"pipeline:task:{task.task_id}",
                    self.redis_ttl,
                    json.dumps(task.to_dict()),
                )
        except Exception as e:
            logger.error(f"Error saving task to Redis: {e}")


# Global orchestrator instance
_orchestrator: Optional[PipelineOrchestrator] = None


async def get_orchestrator() -> PipelineOrchestrator:
    """Get or create global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
        await _orchestrator.initialize()
    
    return _orchestrator
