"""
Prometheus Metrics - Application metrics for monitoring.

Provides comprehensive metrics for:
- HTTP requests (latency, status codes, throughput)
- Database queries (latency, connection pool)
- Celery tasks (duration, success/failure rates)
- Custom business metrics (videos uploaded, users registered)
"""

from typing import Callable
from functools import wraps
import time

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.routing import Match

from app.core.config import settings


# ============================================================================
# HTTP METRICS
# ============================================================================

# Request counters
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently being processed",
    ["method", "endpoint"],
)

# Request duration histogram
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)

# Request size
http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
)


# ============================================================================
# DATABASE METRICS
# ============================================================================

database_queries_total = Counter(
    "database_queries_total",
    "Total database queries",
    ["operation", "table"],
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Database query duration in seconds",
    ["operation", "table"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

database_connections_active = Gauge(
    "database_connections_active",
    "Number of active database connections",
)

database_connections_idle = Gauge(
    "database_connections_idle",
    "Number of idle database connections",
)


# ============================================================================
# REDIS METRICS
# ============================================================================

redis_operations_total = Counter(
    "redis_operations_total",
    "Total Redis operations",
    ["operation"],
)

redis_operation_duration_seconds = Histogram(
    "redis_operation_duration_seconds",
    "Redis operation duration in seconds",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

redis_cache_hits_total = Counter(
    "redis_cache_hits_total",
    "Total cache hits",
    ["cache_type"],
)

redis_cache_misses_total = Counter(
    "redis_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
)


# ============================================================================
# CELERY METRICS
# ============================================================================

celery_tasks_total = Counter(
    "celery_tasks_total",
    "Total Celery tasks",
    ["task_name", "status"],  # status: success, failure, retry
)

celery_task_duration_seconds = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["task_name"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

celery_queue_length = Gauge(
    "celery_queue_length",
    "Number of tasks waiting in queue",
    ["queue_name"],
)

celery_workers_active = Gauge(
    "celery_workers_active",
    "Number of active Celery workers",
    ["queue_name"],
)


# ============================================================================
# ML METRICS
# ============================================================================

ml_predictions_total = Counter(
    "ml_predictions_total",
    "Total ML predictions",
    ["model_name", "status"],  # status: success, failure, cached
)

ml_prediction_duration_seconds = Histogram(
    "ml_prediction_duration_seconds",
    "ML prediction duration in seconds",
    ["model_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ml_cache_hit_rate = Gauge(
    "ml_cache_hit_rate",
    "ML prediction cache hit rate",
    ["model_name"],
)


# ============================================================================
# BUSINESS METRICS
# ============================================================================

users_registered_total = Counter(
    "users_registered_total",
    "Total users registered",
)

users_active = Gauge(
    "users_active",
    "Number of currently active users",
)

videos_uploaded_total = Counter(
    "videos_uploaded_total",
    "Total videos uploaded",
)

videos_total = Gauge(
    "videos_total",
    "Total number of videos",
    ["status"],  # status: uploaded, processing, processed, failed
)

posts_created_total = Counter(
    "posts_created_total",
    "Total posts created",
)

comments_created_total = Counter(
    "comments_created_total",
    "Total comments created",
)

live_streams_active = Gauge(
    "live_streams_active",
    "Number of active live streams",
)

payments_total = Counter(
    "payments_total",
    "Total payments processed",
    ["type", "status"],  # type: one_time, subscription; status: success, failed
)

revenue_total = Counter(
    "revenue_total_usd",
    "Total revenue in USD",
    ["type"],  # type: subscription, tips, ads, content_sales
)


# ============================================================================
# SYSTEM METRICS
# ============================================================================

app_info = Info(
    "app",
    "Application information",
)

# Set app info
app_info.info({
    "version": settings.VERSION if hasattr(settings, 'VERSION') else "1.0.0",
    "environment": settings.ENVIRONMENT,
    "name": "social-flow-backend",
})


# ============================================================================
# PROMETHEUS MIDDLEWARE
# ============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Collect metrics for each HTTP request."""
        # Get endpoint path
        method = request.method
        endpoint = request.url.path
        
        # Normalize endpoint (replace IDs with placeholders)
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                endpoint = route.path
                break
        
        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        # Record request size
        request_size = int(request.headers.get("content-length", 0))
        http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_size)
        
        # Time the request
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            status_code = response.status_code
            
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)
            
            # Record response size
            response_size = int(response.headers.get("content-length", 0))
            http_response_size_bytes.labels(method=method, endpoint=endpoint).observe(response_size)
            
            return response
            
        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


# ============================================================================
# DECORATOR FOR FUNCTION METRICS
# ============================================================================

def track_time(metric: Histogram, *labels):
    """
    Decorator to track function execution time.
    
    Args:
        metric: Histogram metric to record duration
        *labels: Label values for the metric
        
    Example:
        @track_time(database_query_duration_seconds, "SELECT", "users")
        async def get_user(user_id):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(*labels).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(*labels).observe(duration)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_count(metric: Counter, *labels):
    """
    Decorator to count function calls.
    
    Args:
        metric: Counter metric to increment
        *labels: Label values for the metric
        
    Example:
        @track_count(users_registered_total)
        async def register_user(user_data):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            metric.labels(*labels).inc()
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            metric.labels(*labels).inc()
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

import asyncio

async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns:
        Response with Prometheus metrics in text format
    """
    metrics_output = generate_latest()
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST,
    )


# Example usage
"""
# In your FastAPI app setup
from app.core.metrics import PrometheusMiddleware, metrics_endpoint

app.add_middleware(PrometheusMiddleware)

@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()

# In your service methods
from app.core.metrics import (
    database_query_duration_seconds,
    database_queries_total,
    track_time,
)

@track_time(database_query_duration_seconds, "SELECT", "users")
async def get_user(user_id: int):
    database_queries_total.labels(operation="SELECT", table="users").inc()
    # ... query logic
"""
