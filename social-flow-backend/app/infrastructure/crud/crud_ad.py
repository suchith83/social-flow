"""
CRUD operations for ad models (AdCampaign, Ad, AdImpression, AdClick).
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud.base import CRUDBase
from app.models.ad import (
    AdCampaign,
    Ad,
    AdImpression,
    AdClick,
    CampaignStatus,
)
from app.schemas.base import BaseSchema


class CRUDAdCampaign(CRUDBase[AdCampaign, BaseSchema, BaseSchema]):
    """CRUD operations for AdCampaign model."""

    async def get_by_advertiser(
        self,
        db: AsyncSession,
        *,
        advertiser_id: UUID,
        skip: int = 0,
        limit: int = 100,
        status: Optional[CampaignStatus] = None,
    ) -> List[AdCampaign]:
        """Get campaigns by advertiser."""
        query = select(self.model).where(self.model.advertiser_id == advertiser_id)
        
        if status:
            query = query.where(self.model.status == status)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_active(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AdCampaign]:
        """Get active campaigns."""
        now = datetime.now(timezone.utc)
        
        query = (
            select(self.model)
            .where(
                and_(
                    self.model.status == CampaignStatus.ACTIVE,
                    self.model.start_date <= now,
                    self.model.end_date >= now,
                    self.model.budget > self.model.spent,
                )
            )
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def update_spent(
        self,
        db: AsyncSession,
        *,
        campaign_id: UUID,
        amount: float,
    ) -> Optional[AdCampaign]:
        """Update campaign spent amount."""
        campaign = await self.get(db, campaign_id)
        if not campaign:
            return None
        
        campaign.spent += amount
        
        # Auto-pause if budget exhausted
        if campaign.spent >= campaign.budget:
            campaign.status = CampaignStatus.PAUSED
        
        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)
        return campaign

    async def pause_campaign(
        self,
        db: AsyncSession,
        *,
        campaign_id: UUID,
    ) -> Optional[AdCampaign]:
        """Pause a campaign."""
        campaign = await self.get(db, campaign_id)
        if not campaign:
            return None
        
        campaign.status = CampaignStatus.PAUSED
        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)
        return campaign

    async def resume_campaign(
        self,
        db: AsyncSession,
        *,
        campaign_id: UUID,
    ) -> Optional[AdCampaign]:
        """Resume a paused campaign."""
        campaign = await self.get(db, campaign_id)
        if not campaign:
            return None
        
        if campaign.spent < campaign.budget:
            campaign.status = CampaignStatus.ACTIVE
            db.add(campaign)
            await db.commit()
            await db.refresh(campaign)
        
        return campaign


class CRUDAd(CRUDBase[Ad, BaseSchema, BaseSchema]):
    """CRUD operations for Ad model."""

    async def get_by_campaign(
        self,
        db: AsyncSession,
        *,
        campaign_id: UUID,
        skip: int = 0,
        limit: int = 100,
        is_active: Optional[bool] = None,
    ) -> List[Ad]:
        """Get ads by campaign."""
        query = select(self.model).where(self.model.campaign_id == campaign_id)
        
        if is_active is not None:
            query = query.where(self.model.is_active == is_active)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_active_for_targeting(
        self,
        db: AsyncSession,
        *,
        target_age_min: Optional[int] = None,
        target_age_max: Optional[int] = None,
        target_gender: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Ad]:
        """Get active ads matching targeting criteria."""
        now = datetime.now(timezone.utc)
        
        query = select(self.model).join(
            AdCampaign, AdCampaign.id == self.model.campaign_id
        ).where(
            and_(
                self.model.is_active == True,  # noqa: E712
                self.model.is_approved == True,  # noqa: E712
                AdCampaign.status == CampaignStatus.ACTIVE,
                AdCampaign.start_date <= now,
                AdCampaign.end_date >= now,
                AdCampaign.budget > AdCampaign.spent,
            )
        )
        
        # Apply targeting filters if provided
        if target_age_min:
            query = query.where(self.model.target_age_min <= target_age_min)
        if target_age_max:
            query = query.where(self.model.target_age_max >= target_age_max)
        if target_gender:
            query = query.where(
                (self.model.target_gender == target_gender) |
                (self.model.target_gender.is_(None))
            )
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def increment_impressions(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
    ) -> Optional[Ad]:
        """Increment ad impressions count."""
        ad = await self.get(db, ad_id)
        if not ad:
            return None
        
        ad.impressions += 1
        db.add(ad)
        await db.commit()
        await db.refresh(ad)
        return ad

    async def increment_clicks(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
    ) -> Optional[Ad]:
        """Increment ad clicks count."""
        ad = await self.get(db, ad_id)
        if not ad:
            return None
        
        ad.clicks += 1
        db.add(ad)
        await db.commit()
        await db.refresh(ad)
        return ad

    async def get_performance_stats(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
    ):
        """Get performance statistics for an ad."""
        ad = await self.get(db, ad_id)
        if not ad:
            return None
        
        ctr = (ad.clicks / ad.impressions * 100) if ad.impressions > 0 else 0
        
        return {
            "impressions": ad.impressions,
            "clicks": ad.clicks,
            "ctr": round(ctr, 2),
            "cost_per_click": float(ad.bid_amount),
            "total_cost": float(ad.bid_amount * ad.clicks),
        }


class CRUDAdImpression(CRUDBase[AdImpression, BaseSchema, BaseSchema]):
    """CRUD operations for AdImpression model."""

    async def get_by_ad(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AdImpression]:
        """Get impressions by ad."""
        query = (
            select(self.model)
            .where(self.model.ad_id == ad_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_impressions_count(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Get impression count for an ad."""
        query = select(func.count()).select_from(self.model).where(self.model.ad_id == ad_id)
        
        if start_date:
            query = query.where(self.model.created_at >= start_date)
        if end_date:
            query = query.where(self.model.created_at <= end_date)
        
        result = await db.execute(query)
        return result.scalar_one()

    async def get_unique_users_count(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
    ) -> int:
        """Get count of unique users who viewed the ad."""
        query = (
            select(func.count(func.distinct(self.model.user_id)))
            .select_from(self.model)
            .where(self.model.ad_id == ad_id)
        )
        result = await db.execute(query)
        return result.scalar_one()


class CRUDAdClick(CRUDBase[AdClick, BaseSchema, BaseSchema]):
    """CRUD operations for AdClick model."""

    async def get_by_ad(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[AdClick]:
        """Get clicks by ad."""
        query = (
            select(self.model)
            .where(self.model.ad_id == ad_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_clicks_count(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Get click count for an ad."""
        query = select(func.count()).select_from(self.model).where(self.model.ad_id == ad_id)
        
        if start_date:
            query = query.where(self.model.created_at >= start_date)
        if end_date:
            query = query.where(self.model.created_at <= end_date)
        
        result = await db.execute(query)
        return result.scalar_one()

    async def calculate_ctr(
        self,
        db: AsyncSession,
        *,
        ad_id: UUID,
        days: int = 7,
    ) -> float:
        """Calculate click-through rate for an ad."""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get impressions
        impressions_query = (
            select(func.count())
            .select_from(AdImpression)
            .where(
                and_(
                    AdImpression.ad_id == ad_id,
                    AdImpression.created_at >= since,
                )
            )
        )
        impressions_result = await db.execute(impressions_query)
        impressions = impressions_result.scalar_one()
        
        # Get clicks
        clicks_query = (
            select(func.count())
            .select_from(self.model)
            .where(
                and_(
                    self.model.ad_id == ad_id,
                    self.model.created_at >= since,
                )
            )
        )
        clicks_result = await db.execute(clicks_query)
        clicks = clicks_result.scalar_one()
        
        if impressions == 0:
            return 0.0
        
        return round((clicks / impressions) * 100, 2)


# Create singleton instances
ad_campaign = CRUDAdCampaign(AdCampaign)
ad = CRUDAd(Ad)
ad_impression = CRUDAdImpression(AdImpression)
ad_click = CRUDAdClick(AdClick)
