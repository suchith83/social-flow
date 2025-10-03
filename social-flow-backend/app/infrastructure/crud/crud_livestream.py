"""
CRUD operations for livestream models (LiveStream, StreamChat, StreamDonation, StreamViewer).
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud.base import CRUDBase
from app.models.livestream import (
    LiveStream,
    StreamChat,
    StreamDonation,
    StreamViewer,
    StreamStatus,
)
from app.schemas.base import BaseSchema


class CRUDLiveStream(CRUDBase[LiveStream, BaseSchema, BaseSchema]):
    """CRUD operations for LiveStream model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
        status: Optional[StreamStatus] = None,
    ) -> List[LiveStream]:
        """Get streams by user."""
        query = select(self.model).where(self.model.user_id == user_id)
        
        if status:
            query = query.where(self.model.status == status)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_live_streams(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[LiveStream]:
        """Get all currently live streams."""
        query = (
            select(self.model)
            .where(self.model.status == StreamStatus.LIVE)
            .order_by(self.model.viewer_count.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def start_stream(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
    ) -> Optional[LiveStream]:
        """Start a stream."""
        stream = await self.get(db, stream_id)
        if not stream:
            return None
        
        stream.status = StreamStatus.LIVE
        stream.started_at = datetime.now(timezone.utc)
        db.add(stream)
        await db.commit()
        await db.refresh(stream)
        return stream

    async def end_stream(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
    ) -> Optional[LiveStream]:
        """End a stream."""
        stream = await self.get(db, stream_id)
        if not stream:
            return None
        
        stream.status = StreamStatus.ENDED
        stream.ended_at = datetime.now(timezone.utc)
        db.add(stream)
        await db.commit()
        await db.refresh(stream)
        return stream

    async def increment_viewer_count(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
    ) -> Optional[LiveStream]:
        """Increment stream viewer count."""
        stream = await self.get(db, stream_id)
        if not stream:
            return None
        
        stream.viewer_count += 1
        stream.peak_viewer_count = max(stream.peak_viewer_count, stream.viewer_count)
        db.add(stream)
        await db.commit()
        await db.refresh(stream)
        return stream

    async def decrement_viewer_count(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
    ) -> Optional[LiveStream]:
        """Decrement stream viewer count."""
        stream = await self.get(db, stream_id)
        if not stream:
            return None
        
        stream.viewer_count = max(0, stream.viewer_count - 1)
        db.add(stream)
        await db.commit()
        await db.refresh(stream)
        return stream

    async def add_donation(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        amount: float,
    ) -> Optional[LiveStream]:
        """Add donation amount to stream."""
        stream = await self.get(db, stream_id)
        if not stream:
            return None
        
        stream.total_donations += amount
        db.add(stream)
        await db.commit()
        await db.refresh(stream)
        return stream

    async def get_trending(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[LiveStream]:
        """Get trending live streams."""
        query = (
            select(self.model)
            .where(self.model.status == StreamStatus.LIVE)
            .order_by(
                (self.model.viewer_count * 2 + self.model.total_donations).desc()
            )
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())


class CRUDStreamChat(CRUDBase[StreamChat, BaseSchema, BaseSchema]):
    """CRUD operations for StreamChat model."""

    async def get_by_stream(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[StreamChat]:
        """Get chat messages for a stream."""
        query = (
            select(self.model)
            .where(self.model.stream_id == stream_id)
            .order_by(self.model.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_recent_messages(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        limit: int = 50,
    ) -> List[StreamChat]:
        """Get recent chat messages for a stream."""
        query = (
            select(self.model)
            .where(self.model.stream_id == stream_id)
            .order_by(self.model.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        messages = list(result.scalars().all())
        return list(reversed(messages))  # Return in chronological order

    async def delete_user_messages(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        user_id: UUID,
    ) -> int:
        """Delete all messages from a user in a stream (moderation)."""
        from sqlalchemy import delete as sql_delete
        
        query = sql_delete(self.model).where(
            and_(
                self.model.stream_id == stream_id,
                self.model.user_id == user_id,
            )
        )
        result = await db.execute(query)
        await db.commit()
        return result.rowcount


class CRUDStreamDonation(CRUDBase[StreamDonation, BaseSchema, BaseSchema]):
    """CRUD operations for StreamDonation model."""

    async def get_by_stream(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[StreamDonation]:
        """Get donations for a stream."""
        query = (
            select(self.model)
            .where(self.model.stream_id == stream_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_top_donors(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        limit: int = 10,
    ):
        """Get top donors for a stream."""
        from app.models.user import User
        
        query = (
            select(
                User.id,
                User.username,
                User.avatar_url,
                func.sum(self.model.amount).label("total_donated"),
                func.count(self.model.id).label("donation_count"),
            )
            .join(User, User.id == self.model.user_id)
            .where(self.model.stream_id == stream_id)
            .group_by(User.id, User.username, User.avatar_url)
            .order_by(func.sum(self.model.amount).desc())
            .limit(limit)
        )
        result = await db.execute(query)
        return [
            {
                "user_id": row.id,
                "username": row.username,
                "avatar_url": row.avatar_url,
                "total_donated": float(row.total_donated),
                "donation_count": row.donation_count,
            }
            for row in result.all()
        ]

    async def get_total_donations(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
    ) -> float:
        """Get total donation amount for a stream."""
        query = (
            select(func.sum(self.model.amount))
            .select_from(self.model)
            .where(self.model.stream_id == stream_id)
        )
        result = await db.execute(query)
        total = result.scalar_one_or_none()
        return float(total) if total else 0.0

    async def get_user_donations(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[StreamDonation]:
        """Get all donations by a user."""
        query = (
            select(self.model)
            .where(self.model.user_id == user_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())


class CRUDStreamViewer(CRUDBase[StreamViewer, BaseSchema, BaseSchema]):
    """CRUD operations for StreamViewer model."""

    async def get_by_stream(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[StreamViewer]:
        """Get viewers for a stream."""
        query = (
            select(self.model)
            .where(self.model.stream_id == stream_id)
            .order_by(self.model.joined_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_active_viewers(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
    ) -> List[StreamViewer]:
        """Get currently active viewers for a stream."""
        query = (
            select(self.model)
            .where(
                and_(
                    self.model.stream_id == stream_id,
                    self.model.left_at.is_(None),
                )
            )
            .order_by(self.model.joined_at.asc())
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_viewer_count(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
    ) -> int:
        """Get active viewer count for a stream."""
        query = (
            select(func.count())
            .select_from(self.model)
            .where(
                and_(
                    self.model.stream_id == stream_id,
                    self.model.left_at.is_(None),
                )
            )
        )
        result = await db.execute(query)
        return result.scalar_one()

    async def mark_viewer_left(
        self,
        db: AsyncSession,
        *,
        viewer_id: UUID,
    ) -> Optional[StreamViewer]:
        """Mark a viewer as having left the stream."""
        viewer = await self.get(db, viewer_id)
        if not viewer:
            return None
        
        viewer.left_at = datetime.now(timezone.utc)
        db.add(viewer)
        await db.commit()
        await db.refresh(viewer)
        return viewer

    async def get_or_create_viewer(
        self,
        db: AsyncSession,
        *,
        stream_id: UUID,
        user_id: UUID,
    ) -> StreamViewer:
        """Get existing viewer or create new one."""
        # Check for existing active viewer
        query = select(self.model).where(
            and_(
                self.model.stream_id == stream_id,
                self.model.user_id == user_id,
                self.model.left_at.is_(None),
            )
        )
        result = await db.execute(query)
        viewer = result.scalar_one_or_none()
        
        if viewer:
            return viewer
        
        # Create new viewer record
        viewer = StreamViewer(
            stream_id=stream_id,
            user_id=user_id,
            joined_at=datetime.now(timezone.utc),
        )
        db.add(viewer)
        await db.commit()
        await db.refresh(viewer)
        return viewer


# Create singleton instances
livestream = CRUDLiveStream(LiveStream)
stream_chat = CRUDStreamChat(StreamChat)
stream_donation = CRUDStreamDonation(StreamDonation)
stream_viewer = CRUDStreamViewer(StreamViewer)
