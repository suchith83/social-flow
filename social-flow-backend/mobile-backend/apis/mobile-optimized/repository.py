# Repository layer for optimized DB queries
"""
Repository layer for optimized queries.
This mimics DB fetch with selective fields for lightweight responses.
"""

from sqlalchemy.orm import Session
from sqlalchemy import select
from .models import OptimizedItem
from mobile_backend.apis.shared_models import Content  # hypothetical shared DB model


class MobileOptimizedRepository:

    @staticmethod
    def get_items(db: Session, limit: int, offset: int):
        stmt = (
            select(Content.id, Content.title, Content.short_description, Content.thumbnail_url)
            .offset(offset)
            .limit(limit)
        )
        results = db.execute(stmt).fetchall()
        return [
            OptimizedItem(
                id=row.id,
                title=row.title,
                short_description=row.short_description,
                thumbnail_url=row.thumbnail_url,
            )
            for row in results
        ]

    @staticmethod
    def get_updated_since(db: Session, last_sync_ts: str):
        # Hypothetical delta query
        stmt = (
            select(Content.id, Content.title, Content.short_description, Content.thumbnail_url)
            .where(Content.updated_at > last_sync_ts)
        )
        results = db.execute(stmt).fetchall()
        return [
            OptimizedItem(
                id=row.id,
                title=row.title,
                short_description=row.short_description,
                thumbnail_url=row.thumbnail_url,
            )
            for row in results
        ]

    @staticmethod
    def get_deleted_since(db: Session, last_sync_ts: str):
        stmt = select(Content.id).where(Content.deleted_at > last_sync_ts)
        results = db.execute(stmt).fetchall()
        return [row.id for row in results]
