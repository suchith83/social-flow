"""
Pipeline Scheduler.

Manages scheduled execution of AI pipelines including:
- Cron-like scheduling for recurring tasks
- Daily/weekly/monthly batch jobs
- Automatic cache warming
- Model retraining triggers
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, time, timedelta
from enum import Enum

from app.ml_pipelines.orchestrator import (
    get_orchestrator,
    PipelineType,
)

logger = logging.getLogger(__name__)


class ScheduleFrequency(str, Enum):
    """Schedule frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ScheduledTask:
    """Represents a scheduled pipeline task."""
    
    def __init__(
        self,
        name: str,
        pipeline_type: PipelineType,
        config: Dict[str, Any],
        frequency: ScheduleFrequency,
        schedule_time: Optional[time] = None,
        enabled: bool = True,
    ):
        self.name = name
        self.pipeline_type = pipeline_type
        self.config = config
        self.frequency = frequency
        self.schedule_time = schedule_time or time(2, 0)  # Default 2 AM
        self.enabled = enabled
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self._calculate_next_run()
    
    def _calculate_next_run(self) -> None:
        """Calculate next run time based on frequency."""
        now = datetime.utcnow()
        
        if self.frequency == ScheduleFrequency.HOURLY:
            # Next hour
            self.next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        elif self.frequency == ScheduleFrequency.DAILY:
            # Tomorrow at schedule_time
            next_date = now.date() + timedelta(days=1)
            self.next_run = datetime.combine(next_date, self.schedule_time)
        
        elif self.frequency == ScheduleFrequency.WEEKLY:
            # Next Monday at schedule_time
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_date = now.date() + timedelta(days=days_until_monday)
            self.next_run = datetime.combine(next_date, self.schedule_time)
        
        elif self.frequency == ScheduleFrequency.MONTHLY:
            # First day of next month at schedule_time
            if now.month == 12:
                next_date = datetime(now.year + 1, 1, 1).date()
            else:
                next_date = datetime(now.year, now.month + 1, 1).date()
            self.next_run = datetime.combine(next_date, self.schedule_time)
    
    def should_run(self) -> bool:
        """Check if task should run now."""
        if not self.enabled:
            return False
        
        if self.next_run is None:
            return False
        
        return datetime.utcnow() >= self.next_run
    
    def mark_completed(self) -> None:
        """Mark task as completed and calculate next run."""
        self.last_run = datetime.utcnow()
        self._calculate_next_run()


class PipelineScheduler:
    """
    Scheduler for AI pipeline tasks.
    
    Manages recurring tasks like:
    - Daily recommendation pre-computation
    - Weekly model retraining
    - Hourly cache warming
    - Monthly data cleanup
    """
    
    def __init__(self):
        """Initialize pipeline scheduler."""
        self.scheduled_tasks: List[ScheduledTask] = []
        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Register default scheduled tasks
        self._register_default_tasks()
        
        logger.info("Pipeline scheduler initialized")
    
    def _register_default_tasks(self) -> None:
        """Register default scheduled tasks."""
        # Daily recommendation pre-computation at 2 AM
        self.add_task(
            ScheduledTask(
                name="Daily Recommendation Pre-computation",
                pipeline_type=PipelineType.BATCH_RECOMMENDATION_PRECOMPUTE,
                config={
                    "user_ids": [],  # Will be populated dynamically
                    "algorithms": ["hybrid", "smart"],
                },
                frequency=ScheduleFrequency.DAILY,
                schedule_time=time(2, 0),
            )
        )
        
        # Hourly cache warming
        self.add_task(
            ScheduledTask(
                name="Hourly Cache Warming",
                pipeline_type=PipelineType.CACHE_WARMING,
                config={
                    "cache_types": ["recommendations"],
                    "limit": 500,
                },
                frequency=ScheduleFrequency.HOURLY,
            )
        )
        
        # Weekly data cleanup at Sunday 3 AM
        self.add_task(
            ScheduledTask(
                name="Weekly Data Cleanup",
                pipeline_type=PipelineType.DATA_CLEANUP,
                config={
                    "cleanup_types": ["old_cache", "expired_tasks"],
                },
                frequency=ScheduleFrequency.WEEKLY,
                schedule_time=time(3, 0),
            )
        )
    
    def add_task(self, task: ScheduledTask) -> None:
        """Add a scheduled task."""
        self.scheduled_tasks.append(task)
        logger.info(
            f"Scheduled task added: {task.name} "
            f"({task.frequency.value}, next run: {task.next_run})"
        )
    
    def remove_task(self, task_name: str) -> bool:
        """Remove a scheduled task by name."""
        for i, task in enumerate(self.scheduled_tasks):
            if task.name == task_name:
                self.scheduled_tasks.pop(i)
                logger.info(f"Scheduled task removed: {task_name}")
                return True
        return False
    
    def enable_task(self, task_name: str) -> bool:
        """Enable a scheduled task."""
        for task in self.scheduled_tasks:
            if task.name == task_name:
                task.enabled = True
                logger.info(f"Scheduled task enabled: {task_name}")
                return True
        return False
    
    def disable_task(self, task_name: str) -> bool:
        """Disable a scheduled task."""
        for task in self.scheduled_tasks:
            if task.name == task_name:
                task.enabled = False
                logger.info(f"Scheduled task disabled: {task_name}")
                return True
        return False
    
    async def start(self) -> None:
        """Start the scheduler loop."""
        logger.info("Starting pipeline scheduler")
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self) -> None:
        """Stop the scheduler loop."""
        logger.info("Stopping pipeline scheduler")
        self._shutdown = True
        
        if self._scheduler_task:
            await self._scheduler_task
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")
        
        while not self._shutdown:
            try:
                # Check all scheduled tasks
                for task in self.scheduled_tasks:
                    if task.should_run():
                        await self._execute_scheduled_task(task)
                
                # Sleep for 60 seconds before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(60)
        
        logger.info("Scheduler loop stopped")
    
    async def _execute_scheduled_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task."""
        logger.info(f"Executing scheduled task: {task.name}")
        
        try:
            # Get orchestrator
            orchestrator = await get_orchestrator()
            
            # Prepare config
            config = task.config.copy()
            
            # For recommendation pre-computation, get active users
            if task.pipeline_type == PipelineType.BATCH_RECOMMENDATION_PRECOMPUTE:
                if not config.get("user_ids"):
                    config["user_ids"] = await self._get_active_user_ids(limit=1000)
            
            # Submit task to orchestrator
            task_id = await orchestrator.submit_task(
                pipeline_type=task.pipeline_type,
                name=task.name,
                config=config,
                priority=3,  # Medium priority for scheduled tasks
            )
            
            logger.info(
                f"Scheduled task submitted: {task.name} (task_id={task_id})"
            )
            
            # Mark task as completed
            task.mark_completed()
            
        except Exception as e:
            logger.error(f"Error executing scheduled task {task.name}: {e}", exc_info=True)
    
    async def _get_active_user_ids(self, limit: int = 1000) -> List[str]:
        """Get list of active user IDs."""
        try:
            from app.core.database import get_db
            from sqlalchemy import select, desc
            from app.users.models import User
            
            async for db in get_db():
                stmt = (
                    select(User.id)
                    .where(User.is_active.is_(True))
                    .order_by(desc(User.last_login))
                    .limit(limit)
                )
                
                result = await db.execute(stmt)
                user_ids = [str(row[0]) for row in result.all()]
                
                logger.info(f"Retrieved {len(user_ids)} active user IDs")
                return user_ids
                
        except Exception as e:
            logger.error(f"Error getting active user IDs: {e}", exc_info=True)
            return []
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get status of all scheduled tasks."""
        return {
            "total_tasks": len(self.scheduled_tasks),
            "enabled_tasks": sum(1 for t in self.scheduled_tasks if t.enabled),
            "tasks": [
                {
                    "name": task.name,
                    "pipeline_type": task.pipeline_type.value,
                    "frequency": task.frequency.value,
                    "enabled": task.enabled,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                }
                for task in self.scheduled_tasks
            ],
        }


# Global scheduler instance
_scheduler: Optional[PipelineScheduler] = None


def get_scheduler() -> PipelineScheduler:
    """Get or create global scheduler instance."""
    global _scheduler
    
    if _scheduler is None:
        _scheduler = PipelineScheduler()
    
    return _scheduler
