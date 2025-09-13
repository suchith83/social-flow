# Orchestrator for performance strategies
"""
Performance Manager
Coordinates all performance optimization modules.
"""

from .task_scheduler import TaskScheduler
from .thread_pool import ThreadPool
from .cache_layer import CacheLayer
from .load_balancer import LoadBalancer
from .query_optimizer import QueryOptimizer
from .profiler import Profiler
from .metrics import PerformanceMetrics


class PerformanceManager:
    def __init__(self):
        self.scheduler = TaskScheduler()
        self.thread_pool = ThreadPool()
        self.cache = CacheLayer()
        self.balancer = LoadBalancer()
        self.query_optimizer = QueryOptimizer()
        self.profiler = Profiler()
        self.metrics = PerformanceMetrics()

    def execute_task(self, fn, priority=5):
        """Schedule and run a task with profiling."""
        self.scheduler.add_task(lambda: self.profiler.profile(fn)[0], priority)
        result = self.scheduler.run_next()
        self.metrics.log("task_latency", self.profiler.records[-1])
        return result

    def cached_query(self, query: str, exec_fn):
        """Run a query with optimization."""
        result = self.query_optimizer.run_query(query, exec_fn)
        self.metrics.log("query_latency", self.profiler.profile(exec_fn, query)[1])
        return result

    def allocate_task(self, fn):
        """Allocate task to a node via load balancer."""
        node = self.balancer.select_node()
        future = self.thread_pool.submit(fn)
        result = future.result()
        self.balancer.release_node(node)
        self.metrics.log("node_load", len(self.balancer.node_load))
        return result

    def flush_metrics(self) -> dict:
        return self.metrics.flush()
