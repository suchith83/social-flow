# Social Flow Backend - Observability & Hardening Plan

**Generated:** 2025-10-06  
**Focus:** Structured Logging, Metrics, Graceful Degradation, Monitoring Integration

---

## Structured Logging Strategy

### Log Format (JSON)

```json
{
  "timestamp": "2025-01-06T10:30:45.123Z",
  "level": "INFO",
  "service": "social-flow-backend",
  "component": "auth",
  "event": "user_login",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "correlation_id": "req-abc123",
  "duration_ms": 45,
  "status": "success",
  "metadata": {
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "country": "US"
  }
}
```

### Log Levels

| Level | Usage | Examples |
|-------|-------|----------|
| DEBUG | Development debugging | SQL queries, cache hits, detailed flow |
| INFO | Important business events | User login, video upload, payment success |
| WARNING | Degraded state, non-critical issues | Cache miss, fallback to default, retry attempt |
| ERROR | Recoverable errors | API call failed, DB connection retry, validation error |
| CRITICAL | System failure, data loss | DB unreachable, payment processing failure, security breach |

### Key Events to Log

#### Authentication Events

```python
# app/services/auth_service.py
import structlog

logger = structlog.get_logger()

async def login(username: str, password: str) -> LoginResponse:
    """Login with structured logging."""
    correlation_id = generate_correlation_id()
    start_time = time.time()
    
    try:
        user = await user_repo.get_by_email(username)
        if not user:
            logger.warning(
                "login_failed",
                correlation_id=correlation_id,
                username=username,
                reason="user_not_found",
                duration_ms=int((time.time() - start_time) * 1000)
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if not verify_password(password, user.password_hash):
            logger.warning(
                "login_failed",
                correlation_id=correlation_id,
                user_id=str(user.id),
                username=username,
                reason="invalid_password",
                duration_ms=int((time.time() - start_time) * 1000)
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        tokens = create_tokens(user)
        
        logger.info(
            "login_success",
            correlation_id=correlation_id,
            user_id=str(user.id),
            username=username,
            role=user.role.value,
            duration_ms=int((time.time() - start_time) * 1000)
        )
        
        return tokens
    
    except Exception as e:
        logger.error(
            "login_error",
            correlation_id=correlation_id,
            username=username,
            error=str(e),
            error_type=type(e).__name__,
            duration_ms=int((time.time() - start_time) * 1000)
        )
        raise
```

#### Video Processing Events

```python
# app/services/video_service.py

async def process_video(video_id: UUID) -> None:
    """Process video with comprehensive logging."""
    logger = structlog.get_logger()
    
    logger.info(
        "video_processing_started",
        video_id=str(video_id),
        stage="initiation"
    )
    
    try:
        # Download from S3
        logger.info(
            "video_download_started",
            video_id=str(video_id),
            storage="s3"
        )
        video_path = await storage_service.download(video_id)
        
        # Transcode
        logger.info(
            "video_transcode_started",
            video_id=str(video_id),
            resolutions=["1080p", "720p", "480p"]
        )
        transcoded = await transcoding_service.transcode(video_path)
        
        # Generate thumbnail
        logger.info(
            "thumbnail_generation_started",
            video_id=str(video_id)
        )
        thumbnail = await thumbnail_service.generate(video_path)
        
        # Upload outputs
        logger.info(
            "video_upload_started",
            video_id=str(video_id),
            files_count=len(transcoded) + 1
        )
        await storage_service.upload_batch(transcoded + [thumbnail])
        
        logger.info(
            "video_processing_completed",
            video_id=str(video_id),
            duration_ms=processing_time
        )
    
    except Exception as e:
        logger.error(
            "video_processing_failed",
            video_id=str(video_id),
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

#### Payment Events

```python
# app/services/payment_service.py

async def create_subscription(user_id: UUID, plan: str) -> Subscription:
    """Create subscription with audit logging."""
    logger = structlog.get_logger()
    
    logger.info(
        "subscription_creation_started",
        user_id=str(user_id),
        plan=plan,
        provider="stripe"
    )
    
    try:
        stripe_customer = await get_or_create_stripe_customer(user_id)
        
        subscription = stripe.Subscription.create(
            customer=stripe_customer.id,
            items=[{"price": PLAN_PRICES[plan]}]
        )
        
        db_subscription = await subscription_repo.create({
            "user_id": user_id,
            "stripe_subscription_id": subscription.id,
            "plan": plan,
            "status": "active"
        })
        
        logger.info(
            "subscription_created",
            user_id=str(user_id),
            subscription_id=str(db_subscription.id),
            stripe_subscription_id=subscription.id,
            plan=plan,
            amount=subscription.plan.amount,
            currency=subscription.plan.currency
        )
        
        return db_subscription
    
    except stripe.error.StripeError as e:
        logger.error(
            "subscription_creation_failed",
            user_id=str(user_id),
            plan=plan,
            error=str(e),
            error_code=e.code,
            provider="stripe"
        )
        raise
```

#### ML Pipeline Events

```python
# app/ml_pipelines/orchestrator.py

async def run_recommendation_pipeline(user_id: UUID) -> List[Video]:
    """Run recommendation pipeline with detailed logging."""
    logger = structlog.get_logger()
    
    pipeline_id = generate_pipeline_id()
    
    logger.info(
        "ml_pipeline_started",
        pipeline_id=pipeline_id,
        pipeline_type="recommendations",
        user_id=str(user_id)
    )
    
    try:
        # Fetch user history
        start = time.time()
        history = await get_user_history(user_id)
        logger.debug(
            "user_history_fetched",
            pipeline_id=pipeline_id,
            user_id=str(user_id),
            history_size=len(history),
            duration_ms=int((time.time() - start) * 1000)
        )
        
        # Run inference
        start = time.time()
        predictions = await ml_model.predict(history)
        logger.info(
            "ml_inference_completed",
            pipeline_id=pipeline_id,
            model="recommendation_v2",
            predictions_count=len(predictions),
            duration_ms=int((time.time() - start) * 1000)
        )
        
        # Fetch videos
        recommendations = await video_repo.get_by_ids(predictions)
        
        logger.info(
            "ml_pipeline_completed",
            pipeline_id=pipeline_id,
            user_id=str(user_id),
            recommendations_count=len(recommendations),
            total_duration_ms=int((time.time() - pipeline_start) * 1000)
        )
        
        return recommendations
    
    except Exception as e:
        logger.error(
            "ml_pipeline_failed",
            pipeline_id=pipeline_id,
            user_id=str(user_id),
            error=str(e),
            error_type=type(e).__name__
        )
        # Fall back to simple algorithm
        logger.warning(
            "ml_pipeline_fallback",
            pipeline_id=pipeline_id,
            fallback_strategy="trending_videos"
        )
        return await get_trending_videos()
```

---

## Metrics Collection

### Prometheus Integration

#### Application Metrics

```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Database metrics
db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Database connection pool size',
    ['state']  # active, idle
)

# Cache metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result']  # get/set/delete, hit/miss
)

# Video processing metrics
video_processing_duration_seconds = Histogram(
    'video_processing_duration_seconds',
    'Video processing duration',
    ['stage'],  # download, transcode, thumbnail, upload
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)

video_processing_errors_total = Counter(
    'video_processing_errors_total',
    'Total video processing errors',
    ['stage', 'error_type']
)

# Payment metrics
payment_transactions_total = Counter(
    'payment_transactions_total',
    'Total payment transactions',
    ['type', 'status']  # subscription/payout, success/failed
)

payment_amount_total = Counter(
    'payment_amount_total',
    'Total payment amount in cents',
    ['currency', 'type']
)

# ML metrics
ml_inference_duration_seconds = Histogram(
    'ml_inference_duration_seconds',
    'ML inference duration',
    ['model', 'pipeline'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

ml_predictions_total = Counter(
    'ml_predictions_total',
    'Total ML predictions',
    ['model', 'status']  # success/failed/fallback
)
```

#### Middleware Integration

```python
# app/api/middleware/metrics.py
from starlette.middleware.base import BaseHTTPMiddleware
import time

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics."""
    
    async def dispatch(self, request, call_next):
        # Extract route pattern (not raw path)
        route = request.url.path
        for r in request.app.routes:
            match, child_scope = r.matches(request.scope)
            if match:
                route = r.path
                break
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            http_requests_total.labels(
                method=request.method,
                endpoint=route,
                status_code=response.status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=route
            ).observe(duration)
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            http_requests_total.labels(
                method=request.method,
                endpoint=route,
                status_code=500
            ).inc()
            
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=route
            ).observe(duration)
            
            raise


# app/main.py
from app.api.middleware.metrics import MetricsMiddleware
from prometheus_client import make_asgi_app

app.add_middleware(MetricsMiddleware)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

## Graceful Degradation Strategy

### Feature Flags & Fallbacks

```python
# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Settings with feature flags."""
    
    # Core (always required)
    DATABASE_URL: str
    SECRET_KEY: str
    
    # Optional with graceful degradation
    FEATURE_REDIS_ENABLED: bool = True
    FEATURE_S3_ENABLED: bool = True
    FEATURE_ML_ENABLED: bool = True
    FEATURE_CELERY_ENABLED: bool = True
    
    REDIS_URL: str | None = None
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
```

### Degradation Behavior Matrix

| Feature | Enabled | Disabled Fallback | Impact |
|---------|---------|-------------------|--------|
| **Redis Cache** | Cache hits, faster responses | Direct DB queries | +50-200ms latency |
| **S3 Storage** | Cloud storage, CDN | Local filesystem | Limited scalability |
| **ML Recommendations** | Personalized content | Trending/Popular | Generic content |
| **Celery Tasks** | Async processing | Synchronous execution | Blocking operations |
| **2FA** | Enhanced security | Password-only auth | Reduced security |
| **Live Streaming** | Real-time video | VOD only | No live content |

### Implementation Examples

#### Cache Degradation

```python
# app/infrastructure/cache.py

class CacheService:
    """Cache service with graceful degradation."""
    
    def __init__(self, redis_client: Redis | None = None):
        self.redis = redis_client
        self.enabled = redis_client is not None
    
    async def get(self, key: str) -> Any | None:
        """Get from cache with fallback."""
        if not self.enabled:
            logger.debug("cache_disabled", operation="get", key=key)
            return None
        
        try:
            value = await self.redis.get(key)
            cache_operations_total.labels(
                operation="get",
                result="hit" if value else "miss"
            ).inc()
            return value
        except Exception as e:
            logger.warning(
                "cache_operation_failed",
                operation="get",
                key=key,
                error=str(e)
            )
            cache_operations_total.labels(
                operation="get",
                result="error"
            ).inc()
            return None  # Graceful degradation
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set cache with fallback."""
        if not self.enabled:
            logger.debug("cache_disabled", operation="set", key=key)
            return False
        
        try:
            await self.redis.setex(key, ttl, value)
            cache_operations_total.labels(
                operation="set",
                result="success"
            ).inc()
            return True
        except Exception as e:
            logger.warning(
                "cache_operation_failed",
                operation="set",
                key=key,
                error=str(e)
            )
            return False  # Don't raise, just skip caching


# Usage in service
async def get_video(video_id: UUID) -> Video:
    """Get video with cache fallback."""
    # Try cache first
    cache_key = f"video:{video_id}"
    cached = await cache_service.get(cache_key)
    if cached:
        return Video.parse_raw(cached)
    
    # Fallback to DB
    video = await video_repo.get(video_id)
    
    # Try to cache (fire and forget)
    await cache_service.set(cache_key, video.json())
    
    return video
```

#### Storage Degradation

```python
# app/services/storage_service.py

class StorageService:
    """Storage service with S3/local fallback."""
    
    def __init__(self, settings: Settings):
        self.s3_enabled = settings.FEATURE_S3_ENABLED
        if self.s3_enabled:
            self.s3_client = boto3.client('s3')
            self.bucket = settings.S3_BUCKET_NAME
        else:
            self.local_path = Path("./local_storage")
            self.local_path.mkdir(exist_ok=True)
            logger.warning(
                "storage_degraded",
                mode="local_filesystem",
                reason="s3_disabled"
            )
    
    async def upload(self, file_path: Path, key: str) -> str:
        """Upload with S3/local fallback."""
        if self.s3_enabled:
            try:
                await self.s3_client.upload_file(str(file_path), self.bucket, key)
                url = f"https://{self.bucket}.s3.amazonaws.com/{key}"
                logger.info("storage_upload_success", key=key, storage="s3")
                return url
            except Exception as e:
                logger.error(
                    "storage_upload_failed",
                    key=key,
                    storage="s3",
                    error=str(e)
                )
                raise
        else:
            # Local filesystem fallback
            local_file = self.local_path / key
            local_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, local_file)
            url = f"/static/uploads/{key}"
            logger.info("storage_upload_success", key=key, storage="local")
            return url
```

#### ML Degradation

```python
# app/services/recommendation_service.py

class RecommendationService:
    """Recommendation service with ML/simple fallback."""
    
    def __init__(self, ml_enabled: bool = True):
        self.ml_enabled = ml_enabled
        if ml_enabled:
            try:
                self.model = load_ml_model()
            except Exception as e:
                logger.error("ml_model_load_failed", error=str(e))
                self.ml_enabled = False
    
    async def get_recommendations(self, user_id: UUID, limit: int = 20) -> List[Video]:
        """Get recommendations with ML/trending fallback."""
        if self.ml_enabled:
            try:
                start = time.time()
                predictions = await self.model.predict(user_id)
                duration = time.time() - start
                
                ml_inference_duration_seconds.labels(
                    model="recommendation_v2",
                    pipeline="user_recommendations"
                ).observe(duration)
                
                ml_predictions_total.labels(
                    model="recommendation_v2",
                    status="success"
                ).inc()
                
                logger.info(
                    "recommendations_ml",
                    user_id=str(user_id),
                    count=len(predictions),
                    duration_ms=int(duration * 1000)
                )
                
                return predictions[:limit]
            
            except Exception as e:
                logger.error(
                    "ml_recommendation_failed",
                    user_id=str(user_id),
                    error=str(e)
                )
                ml_predictions_total.labels(
                    model="recommendation_v2",
                    status="failed"
                ).inc()
                # Fall through to simple algorithm
        
        # Fallback: trending videos
        logger.info(
            "recommendations_fallback",
            user_id=str(user_id),
            strategy="trending"
        )
        ml_predictions_total.labels(
            model="recommendation_v2",
            status="fallback"
        ).inc()
        
        return await video_repo.get_trending(limit=limit)
```

---

## Health Check Hardening

### Parallel Health Checks

```python
# app/api/v1/endpoints/health.py
import asyncio
from typing import Dict, Any

@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Comprehensive health check with parallel execution."""
    start_time = time.time()
    
    # Run all checks in parallel with timeout
    results = await asyncio.gather(
        check_database(),
        check_redis(),
        check_s3(),
        check_celery(),
        check_ml(),
        return_exceptions=True  # Don't fail on individual errors
    )
    
    # Parse results
    db_healthy, redis_healthy, s3_healthy, celery_healthy, ml_healthy = results
    
    # Convert exceptions to status
    def parse_result(result, name: str) -> Dict[str, Any]:
        if isinstance(result, Exception):
            logger.warning(
                "health_check_failed",
                component=name,
                error=str(result)
            )
            return {
                "status": "unhealthy",
                "error": str(result),
                "error_type": type(result).__name__
            }
        return result
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "duration_ms": int((time.time() - start_time) * 1000),
        "components": {
            "database": parse_result(db_healthy, "database"),
            "redis": parse_result(redis_healthy, "redis"),
            "s3": parse_result(s3_healthy, "s3"),
            "celery": parse_result(celery_healthy, "celery"),
            "ml": parse_result(ml_healthy, "ml")
        }
    }
    
    # Overall status: degraded if any non-critical component fails
    critical_components = ["database"]
    optional_components = ["redis", "s3", "celery", "ml"]
    
    critical_failures = [
        name for name in critical_components
        if health_status["components"][name]["status"] == "unhealthy"
    ]
    
    optional_failures = [
        name for name in optional_components
        if health_status["components"][name]["status"] == "unhealthy"
    ]
    
    if critical_failures:
        health_status["status"] = "unhealthy"
        health_status["critical_failures"] = critical_failures
    elif optional_failures:
        health_status["status"] = "degraded"
        health_status["optional_failures"] = optional_failures
    
    return health_status


async def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        async with get_db() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        return {
            "status": "healthy",
            "response_time_ms": 5
        }
    except Exception as e:
        raise Exception(f"Database check failed: {str(e)}")


async def check_redis() -> Dict[str, Any]:
    """Check Redis connectivity."""
    if not settings.FEATURE_REDIS_ENABLED:
        return {"status": "disabled"}
    
    try:
        await redis_client.ping()
        return {
            "status": "healthy",
            "response_time_ms": 2
        }
    except Exception as e:
        raise Exception(f"Redis check failed: {str(e)}")


async def check_s3() -> Dict[str, Any]:
    """Check S3 connectivity."""
    if not settings.FEATURE_S3_ENABLED:
        return {"status": "disabled"}
    
    try:
        s3_client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
        return {
            "status": "healthy",
            "response_time_ms": 50
        }
    except Exception as e:
        raise Exception(f"S3 check failed: {str(e)}")
```

---

## Monitoring Integration

### Grafana Dashboard (JSON)

```json
{
  "dashboard": {
    "title": "Social Flow Backend Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (endpoint, status_code)"
          }
        ]
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status_code=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))"
          }
        ]
      },
      {
        "title": "Database Connection Pool",
        "targets": [
          {
            "expr": "db_connection_pool_size"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "sum(rate(cache_operations_total{result=\"hit\"}[5m])) / sum(rate(cache_operations_total{operation=\"get\"}[5m]))"
          }
        ]
      },
      {
        "title": "Video Processing Duration",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(video_processing_duration_seconds_bucket[5m])) by (stage)"
          }
        ]
      },
      {
        "title": "ML Inference Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_inference_duration_seconds_bucket[5m])) by (model)"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: social_flow_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: DatabaseDown
        expr: up{job="social-flow-backend"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time (p95 > 2s)"
      
      - alert: CacheDown
        expr: redis_up == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis cache is down (degraded mode)"
      
      - alert: VideoProcessingBacklog
        expr: celery_queue_length{queue="video_processing"} > 100
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Video processing backlog growing"
```

---

## Summary

### Implementation Priority

1. **Immediate (Week 1)**
   - Add structured logging to auth, payment, video services
   - Implement parallel health checks with graceful degradation
   - Add Prometheus metrics endpoint

2. **Short-term (Week 2-3)**
   - Complete metrics coverage (DB, cache, ML, Celery)
   - Create Grafana dashboards
   - Set up alerting rules

3. **Medium-term (Week 4-6)**
   - Implement graceful degradation for all optional services
   - Add correlation IDs for request tracing
   - Set up centralized logging (ELK/Loki)

4. **Ongoing**
   - Monitor and tune alert thresholds
   - Add custom metrics for business KPIs
   - Review and improve log quality

### Success Metrics

- **Observability:** All critical paths have structured logging
- **Performance:** p95 latency < 500ms for API endpoints
- **Reliability:** 99.9% uptime (degraded mode counts as up)
- **Resilience:** System continues operating when Redis/S3/ML unavailable
- **Monitoring:** 100% coverage of health endpoints, metrics, alerts

---

**All audit deliverables complete. Ready for implementation.**
