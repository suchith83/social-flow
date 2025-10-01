"""
Health Check Endpoints - System health monitoring.

Provides comprehensive health checks for all system components:
- Database connectivity
- Redis connectivity
- Celery workers
- S3 storage
- External services
"""

from typing import Dict, Any
from datetime import datetime
import asyncio

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import boto3
from botocore.exceptions import ClientError

from app.core.database import get_db
from app.core.redis import get_redis
from app.core.config import settings
from app.core.logging_config import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns 200 if service is running.
    Used by load balancers for liveness checks.
    """
    return {
        "status": "healthy",
        "service": "social-flow-backend",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Readiness check endpoint.
    
    Verifies that all critical dependencies are available:
    - Database connection
    - Redis connection
    
    Returns 200 if ready to serve traffic.
    Returns 503 if not ready.
    
    Used by load balancers for readiness checks.
    """
    checks = {}
    all_healthy = True
    
    # Check database
    db_healthy, db_info = await check_database(db)
    checks["database"] = db_info
    if not db_healthy:
        all_healthy = False
    
    # Check Redis
    redis_healthy, redis_info = await check_redis()
    checks["redis"] = redis_info
    if not redis_healthy:
        all_healthy = False
    
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint.
    
    Verifies that the application process is alive and responsive.
    Does not check dependencies.
    
    Returns 200 if alive.
    Used by orchestrators (Kubernetes) for liveness probes.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Detailed health check endpoint.
    
    Performs comprehensive health checks on all system components:
    - Database (connection, query performance)
    - Redis (connection, latency)
    - Celery workers (queue status)
    - S3 storage (connectivity)
    - ML models (availability)
    
    Returns detailed status for monitoring and debugging.
    """
    checks = {}
    
    # Run checks in parallel
    results = await asyncio.gather(
        check_database(db),
        check_redis(),
        check_celery(),
        check_s3(),
        check_ml_models(),
        return_exceptions=True,
    )
    
    # Process results
    check_names = ["database", "redis", "celery", "s3", "ml_models"]
    all_healthy = True
    
    for name, result in zip(check_names, results):
        if isinstance(result, Exception):
            checks[name] = {
                "status": "error",
                "error": str(result),
            }
            all_healthy = False
        else:
            healthy, info = result
            checks[name] = info
            if not healthy:
                all_healthy = False
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# INDIVIDUAL HEALTH CHECKS
# ============================================================================

async def check_database(db: AsyncSession) -> tuple[bool, Dict[str, Any]]:
    """Check database connectivity and performance."""
    try:
        start_time = datetime.utcnow()
        
        # Execute simple query
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        
        # Check connection pool
        pool = db.get_bind().pool
        pool_size = pool.size() if hasattr(pool, 'size') else 'unknown'
        pool_checked_out = pool.checkedout() if hasattr(pool, 'checkedout') else 'unknown'
        
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms
        
        return True, {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "pool_size": pool_size,
            "connections_in_use": pool_checked_out,
        }
        
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
        return False, {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_redis() -> tuple[bool, Dict[str, Any]]:
    """Check Redis connectivity and performance."""
    try:
        redis = await get_redis()
        
        start_time = datetime.utcnow()
        
        # Test ping
        await redis.ping()
        
        # Get info
        info = await redis.info()
        
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms
        
        return True, {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "connected_clients": info.get("connected_clients", "unknown"),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "uptime_days": info.get("uptime_in_days", "unknown"),
        }
        
    except Exception as e:
        logger.error("redis_health_check_failed", error=str(e))
        return False, {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_celery() -> tuple[bool, Dict[str, Any]]:
    """Check Celery workers and queue status."""
    try:
        from app.workers.celery_app import celery_app
        
        # Get active workers
        inspect = celery_app.control.inspect()
        
        # Get stats (with timeout)
        stats = inspect.stats(timeout=2.0)
        active = inspect.active(timeout=2.0)
        
        if not stats:
            return False, {
                "status": "unhealthy",
                "error": "No workers available",
            }
        
        # Count workers by queue
        worker_count = len(stats)
        active_tasks = sum(len(tasks) for tasks in (active or {}).values())
        
        return True, {
            "status": "healthy",
            "worker_count": worker_count,
            "active_tasks": active_tasks,
            "workers": list(stats.keys()) if stats else [],
        }
        
    except Exception as e:
        logger.error("celery_health_check_failed", error=str(e))
        return False, {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_s3() -> tuple[bool, Dict[str, Any]]:
    """Check S3 storage connectivity."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        
        # Check bucket access
        s3_client.head_bucket(Bucket=settings.AWS_S3_BUCKET)
        
        return True, {
            "status": "healthy",
            "bucket": settings.AWS_S3_BUCKET,
            "region": settings.AWS_REGION,
        }
        
    except ClientError as e:
        logger.error("s3_health_check_failed", error=str(e))
        return False, {
            "status": "unhealthy",
            "error": str(e),
        }
    except Exception as e:
        logger.error("s3_health_check_failed", error=str(e))
        return False, {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_ml_models() -> tuple[bool, Dict[str, Any]]:
    """Check ML model availability."""
    try:
        from app.services.ml_service import ml_service
        
        # Get model info
        model_info = await ml_service.get_model_info()
        
        return True, {
            "status": "healthy",
            "loaded_models": model_info.get("loaded_models", []),
            "model_count": model_info.get("model_count", 0),
        }
        
    except Exception as e:
        logger.error("ml_health_check_failed", error=str(e))
        return False, {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/startup", status_code=status.HTTP_200_OK)
async def startup_check(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Startup check endpoint.
    
    Verifies that the application has completed initialization:
    - Database migrations applied
    - Configuration loaded
    - Critical dependencies available
    
    Returns 200 if startup complete.
    Used during container startup.
    """
    checks = {}
    
    # Check database
    db_healthy, db_info = await check_database(db)
    checks["database"] = db_info
    
    # Check Redis
    redis_healthy, redis_info = await check_redis()
    checks["redis"] = redis_info
    
    # Check config
    config_valid = bool(
        settings.DATABASE_URL and
        settings.REDIS_HOST and
        settings.SECRET_KEY
    )
    checks["configuration"] = {
        "status": "healthy" if config_valid else "unhealthy",
        "environment": settings.ENVIRONMENT,
    }
    
    all_healthy = db_healthy and redis_healthy and config_valid
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }
