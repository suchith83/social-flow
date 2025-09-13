# Priority-based task scheduling
"""
Priority-Based Task Scheduler
Executes tasks based on priority with fair queuing.
"""

import heapq
import time
from typing import Callable, Any, List, Tuple
from .config import CONFIG


class TaskScheduler:
    def __init__(self):
        self.queue: List[Tuple[int, float, Callable]] = []

    def add_task(self, task_fn: Callable, priority: int = CONFIG.default_priority):
        """Add a task with a given priority (lower = higher priority)."""
        timestamp = time.time()
        heapq.heappush(self.queue, (priority, timestamp, task_fn))

    def run_next(self) -> Any:
        """Run the highest-priority task."""
        if not self.queue:
            return None
        _, _, task_fn = heapq.heappop(self.queue)
        return task_fn()

    def run_all(self) -> List[Any]:
        """Run all tasks in priority order."""
        results = []
        while self.queue:
            results.append(self.run_next())
        return results
