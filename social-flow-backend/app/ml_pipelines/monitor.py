"""
Pipeline Monitor.

Monitors AI pipeline execution including:
- Task execution metrics
- Performance tracking
- Error rate monitoring
- Resource utilization
- Health checks
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json

from app.core.redis import get_redis

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """
    Monitor for AI pipeline execution.
    
    Tracks metrics, performance, and health of pipeline tasks.
    """
    
    def __init__(
        self,
        metrics_retention_days: int = 7,
    ):
        """
        Initialize pipeline monitor.
        
        Args:
            metrics_retention_days: Number of days to retain metrics
        """
        self.metrics_retention_days = metrics_retention_days
        
        # In-memory metrics (recent)
        self.task_counts: Dict[str, int] = defaultdict(int)
        self.task_durations: Dict[str, List[float]] = defaultdict(list)
        self.task_errors: List[Dict[str, Any]] = []
        
        logger.info("Pipeline monitor initialized")
    
    async def record_task_execution(self, task: Any) -> None:
        """
        Record task execution metrics.
        
        Args:
            task: PipelineTask instance
        """
        try:
            pipeline_type = task.pipeline_type.value
            status = task.status.value
            
            # Update counts
            self.task_counts[f"{pipeline_type}_{status}"] += 1
            
            # Record duration
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                self.task_durations[pipeline_type].append(duration)
                
                # Keep only last 100 durations per type
                if len(self.task_durations[pipeline_type]) > 100:
                    self.task_durations[pipeline_type] = self.task_durations[pipeline_type][-100:]
            
            # Record errors
            if task.error:
                self.task_errors.append({
                    "task_id": task.task_id,
                    "pipeline_type": pipeline_type,
                    "error": task.error,
                    "timestamp": task.completed_at.isoformat() if task.completed_at else None,
                })
                
                # Keep only last 50 errors
                if len(self.task_errors) > 50:
                    self.task_errors = self.task_errors[-50:]
            
            # Save to Redis for persistence
            await self._save_metrics_to_redis(task)
            
        except Exception as e:
            logger.error(f"Error recording task execution: {e}", exc_info=True)
    
    async def _save_metrics_to_redis(self, task: Any) -> None:
        """Save task metrics to Redis."""
        try:
            redis = await get_redis()
            if not redis:
                return
            
            # Save task execution record
            record_key = f"pipeline:metrics:task:{task.task_id}"
            record_data = {
                "pipeline_type": task.pipeline_type.value,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "duration": (task.completed_at - task.started_at).total_seconds() 
                           if task.started_at and task.completed_at else None,
                "error": task.error,
            }
            
            await redis.setex(
                record_key,
                self.metrics_retention_days * 86400,
                json.dumps(record_data),
            )
            
            # Update aggregated metrics
            date_key = datetime.utcnow().strftime("%Y-%m-%d")
            metrics_key = f"pipeline:metrics:daily:{date_key}"
            
            # Increment counters
            await redis.hincrby(metrics_key, f"{task.pipeline_type.value}_total", 1)
            await redis.hincrby(metrics_key, f"{task.pipeline_type.value}_{task.status.value}", 1)
            
            # Set expiry on daily metrics
            await redis.expire(metrics_key, self.metrics_retention_days * 86400)
            
        except Exception as e:
            logger.error(f"Error saving metrics to Redis: {e}", exc_info=True)
    
    async def get_metrics(
        self,
        pipeline_type: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get pipeline metrics.
        
        Args:
            pipeline_type: Filter by pipeline type (None = all)
            days: Number of days to retrieve
            
        Returns:
            Dict with metrics
        """
        try:
            redis = await get_redis()
            if not redis:
                return self._get_in_memory_metrics(pipeline_type)
            
            # Get daily metrics from Redis
            daily_metrics = []
            for i in range(days):
                date = datetime.utcnow().date() - timedelta(days=i)
                date_key = date.strftime("%Y-%m-%d")
                metrics_key = f"pipeline:metrics:daily:{date_key}"
                
                metrics = await redis.hgetall(metrics_key)
                if metrics:
                    daily_metrics.append({
                        "date": date_key,
                        "metrics": metrics,
                    })
            
            # Combine with in-memory metrics
            return {
                "daily_metrics": daily_metrics,
                "recent_metrics": self._get_in_memory_metrics(pipeline_type),
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}", exc_info=True)
            return self._get_in_memory_metrics(pipeline_type)
    
    def _get_in_memory_metrics(self, pipeline_type: Optional[str] = None) -> Dict[str, Any]:
        """Get in-memory metrics."""
        # Filter by pipeline type if specified
        if pipeline_type:
            counts = {k: v for k, v in self.task_counts.items() if k.startswith(pipeline_type)}
            durations = {k: v for k, v in self.task_durations.items() if k == pipeline_type}
            errors = [e for e in self.task_errors if e["pipeline_type"] == pipeline_type]
        else:
            counts = dict(self.task_counts)
            durations = dict(self.task_durations)
            errors = list(self.task_errors)
        
        # Calculate statistics
        stats = {}
        for pt, duration_list in durations.items():
            if duration_list:
                stats[pt] = {
                    "avg_duration": sum(duration_list) / len(duration_list),
                    "min_duration": min(duration_list),
                    "max_duration": max(duration_list),
                    "total_executions": len(duration_list),
                }
        
        return {
            "task_counts": counts,
            "duration_stats": stats,
            "recent_errors": errors[-10:],  # Last 10 errors
            "total_errors": len(errors),
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of pipelines.
        
        Returns:
            Dict with health metrics
        """
        try:
            metrics = await self.get_metrics(days=1)
            recent = metrics.get("recent_metrics", {})
            
            # Calculate error rate
            total_tasks = sum(recent.get("task_counts", {}).values())
            total_errors = recent.get("total_errors", 0)
            error_rate = (total_errors / total_tasks * 100) if total_tasks > 0 else 0
            
            # Determine health status
            if error_rate > 20:
                health = "critical"
            elif error_rate > 10:
                health = "degraded"
            elif error_rate > 5:
                health = "warning"
            else:
                health = "healthy"
            
            return {
                "status": health,
                "error_rate": error_rate,
                "total_tasks_24h": total_tasks,
                "total_errors_24h": total_errors,
                "duration_stats": recent.get("duration_stats", {}),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}", exc_info=True)
            return {
                "status": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report.
        
        Returns:
            Dict with performance metrics
        """
        try:
            metrics = await self.get_metrics(days=7)
            
            # Analyze trends
            daily_data = metrics.get("daily_metrics", [])
            
            if not daily_data:
                return {"message": "No data available"}
            
            # Calculate totals
            total_tasks = 0
            total_errors = 0
            
            for day_data in daily_data:
                day_metrics = day_data.get("metrics", {})
                for key, value in day_metrics.items():
                    if key.endswith("_total"):
                        total_tasks += int(value)
                    elif key.endswith("_failed"):
                        total_errors += int(value)
            
            return {
                "period_days": 7,
                "total_tasks": total_tasks,
                "total_errors": total_errors,
                "error_rate": (total_errors / total_tasks * 100) if total_tasks > 0 else 0,
                "avg_tasks_per_day": total_tasks / 7,
                "daily_breakdown": daily_data,
                "duration_stats": metrics.get("recent_metrics", {}).get("duration_stats", {}),
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}", exc_info=True)
            return {"error": str(e)}
