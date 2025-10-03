"""
Infrastructure Layer Package.

Contains all infrastructure implementations:
- Storage (S3, Azure, GCS)
- Caching (Redis)
- Messaging (Celery, RabbitMQ)
- External Services (Payment, ML, Email)
- Repositories (Data Access Layer)
"""

from app.infrastructure.storage import (
    StorageManager,
    get_storage_manager,
    initialize_storage,
    StorageProvider,
)

from app.infrastructure.repositories import (
    UserRepository,
    VideoRepository,
    PostRepository,
)

__all__ = [
    # Storage
    "StorageManager",
    "get_storage_manager",
    "initialize_storage",
    "StorageProvider",
    # Repositories
    "UserRepository",
    "VideoRepository",
    "PostRepository",
]
