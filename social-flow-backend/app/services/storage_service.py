"""
Storage Service - Backward Compatibility Module

DEPRECATED: This module is maintained for backward compatibility only.
All new code should use app.infrastructure.storage.manager directly.

This import redirection will be removed in Phase 3 after all services
have been migrated to the new unified storage infrastructure.
"""

import warnings

# Import from legacy wrapper
from app.services.storage_service_legacy import (
    StorageService,
    storage_service
)

# Deprecation warning
warnings.warn(
    "Importing from app.services.storage_service is deprecated. "
    "Use app.infrastructure.storage.manager.get_storage_manager() instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = ["StorageService", "storage_service"]

