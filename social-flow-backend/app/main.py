"""
Main FastAPI application entry point.

This module initializes the FastAPI application with all necessary middleware,
routers, and configurations for the Social Flow backend.
"""

from contextlib import asynccontextmanager
import os
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.application.container import get_container
from app.core.middleware import RequestContextMiddleware
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.redis import init_redis
from app.api.v1.router import api_router
from app.core.exceptions import SocialFlowException


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    setup_logging()
    
    # Skip database initialization in test mode (tests manage their own database)
    if not settings.TESTING:
        await init_db()
        await init_redis()
        
        # Initialize unified storage infrastructure
        from app.infrastructure.storage import initialize_storage
        try:
            await initialize_storage()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize storage infrastructure: {e}")
            logger.warning("Application will continue but storage functionality may be limited")
        
        # Initialize AI Pipeline Orchestrator and Scheduler
        try:
            from app.ml_pipelines.orchestrator import get_orchestrator
            from app.ml_pipelines.scheduler import get_scheduler
            
            # Initialize orchestrator singleton (assignment ensures initialization)
            _ = await get_orchestrator()
            scheduler = get_scheduler()
            await scheduler.start()
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info("AI Pipeline Orchestrator and Scheduler initialized successfully")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize AI Pipeline services: {e}")
            logger.warning("Application will continue but AI pipeline functionality may be limited")
    
    yield
    
    # Shutdown
    # Cleanup AI Pipeline resources
    if not settings.TESTING:
        try:
            from app.ml_pipelines.scheduler import get_scheduler
            scheduler = get_scheduler()
            await scheduler.stop()
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info("AI Pipeline services shut down gracefully")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error during AI Pipeline shutdown: {e}")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan,
    )
    
    # Security middleware
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Trusted host middleware
    if settings.ALLOWED_HOSTS:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )
    
    # Attach container early (makes it discoverable in dependency graph / tests)
    app.state.container = get_container()

    # Request context middleware (request id, latency)
    app.add_middleware(RequestContextMiddleware)  # type: ignore[arg-type]

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    # Exception handlers
    @app.exception_handler(SocialFlowException)
    async def social_flow_exception_handler(request, exc: SocialFlowException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {"status": "healthy", "version": settings.VERSION}
    
    # Metrics endpoint
    if settings.ENABLE_METRICS:
        Instrumentator().instrument(app).expose(app)
    
    return app


# Create the application instance
app = create_application()

if __name__ == "__main__":
    import uvicorn
    
    # Bind to localhost by default for safety; allow override via env for containers
    server_host = os.getenv("HOST", "127.0.0.1")
    server_port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app.main:app",
        host=server_host,
        port=server_port,
        reload=settings.DEBUG,
        log_level="info",
    )
