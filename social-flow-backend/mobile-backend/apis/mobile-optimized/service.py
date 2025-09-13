# Service layer (caching, delta sync logic)
"""
Service layer for mobile-optimized endpoints.
Encapsulates caching and business logic.
"""

from sqlalchemy.orm import Session
from datetime import datetime
from .repository import MobileOptimizedRepository
from .models import DeltaSyncRequest, DeltaSyncResponse
from .config import get_config
from .utils import cache_result

config = get_config()


class MobileOptimizedService:

    @staticmethod
    @cache_result(ttl=config.CACHE_TTL)
    def list_items(db: Session, limit: int, offset: int):
        items = MobileOptimizedRepository.get_items(db, limit, offset)
        return {
            "items": items,
            "next_offset": offset + len(items) if len(items) == limit else None,
            "has_more": len(items) == limit,
        }

    @staticmethod
    def delta_sync(db: Session, sync_request: DeltaSyncRequest):
        updated = MobileOptimizedRepository.get_updated_since(db, sync_request.last_sync_timestamp)
        deleted = MobileOptimizedRepository.get_deleted_since(db, sync_request.last_sync_timestamp)
        return DeltaSyncResponse(
            updated_items=updated,
            deleted_item_ids=deleted,
            server_timestamp=datetime.utcnow().isoformat()
        )
