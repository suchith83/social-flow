"""
CRUD operations for payment models (Payment, Subscription, Payout, Transaction).
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud.base import CRUDBase
from app.models.payment import (
    Payment,
    Subscription,
    Payout,
    Transaction,
    PaymentStatus,
    SubscriptionStatus,
    PayoutStatus,
)
from app.schemas.base import BaseSchema


class CRUDPayment(CRUDBase[Payment, BaseSchema, BaseSchema]):
    """CRUD operations for Payment model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
        status: Optional[PaymentStatus] = None,
    ) -> List[Payment]:
        """Get payments by user."""
        query = select(self.model).where(self.model.user_id == user_id)
        
        if status:
            query = query.where(self.model.status == status)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_by_stripe_payment_intent(
        self,
        db: AsyncSession,
        *,
        payment_intent_id: str,
    ) -> Optional[Payment]:
        """Get payment by Stripe payment intent ID."""
        return await self.get_by_field(db, "stripe_payment_intent_id", payment_intent_id)

    async def update_status(
        self,
        db: AsyncSession,
        *,
        payment_id: UUID,
        status: PaymentStatus,
    ) -> Optional[Payment]:
        """Update payment status."""
        payment = await self.get(db, payment_id)
        if not payment:
            return None
        
        payment.status = status
        db.add(payment)
        await db.commit()
        await db.refresh(payment)
        return payment

    async def get_total_revenue(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> float:
        """Calculate total revenue from successful payments."""
        query = select(func.sum(self.model.amount)).where(
            self.model.status == PaymentStatus.SUCCEEDED
        )
        
        if user_id:
            query = query.where(self.model.user_id == user_id)
        if start_date:
            query = query.where(self.model.created_at >= start_date)
        if end_date:
            query = query.where(self.model.created_at <= end_date)
        
        result = await db.execute(query)
        total = result.scalar_one_or_none()
        return float(total) if total else 0.0


class CRUDSubscription(CRUDBase[Subscription, BaseSchema, BaseSchema]):
    """CRUD operations for Subscription model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Subscription]:
        """Get subscriptions by user."""
        query = (
            select(self.model)
            .where(self.model.user_id == user_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_active_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> Optional[Subscription]:
        """Get active subscription for user."""
        query = select(self.model).where(
            and_(
                self.model.user_id == user_id,
                self.model.status == SubscriptionStatus.ACTIVE,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_stripe_subscription(
        self,
        db: AsyncSession,
        *,
        stripe_subscription_id: str,
    ) -> Optional[Subscription]:
        """Get subscription by Stripe subscription ID."""
        return await self.get_by_field(db, "stripe_subscription_id", stripe_subscription_id)

    async def update_status(
        self,
        db: AsyncSession,
        *,
        subscription_id: UUID,
        status: SubscriptionStatus,
    ) -> Optional[Subscription]:
        """Update subscription status."""
        subscription = await self.get(db, subscription_id)
        if not subscription:
            return None
        
        subscription.status = status
        db.add(subscription)
        await db.commit()
        await db.refresh(subscription)
        return subscription

    async def cancel_subscription(
        self,
        db: AsyncSession,
        *,
        subscription_id: UUID,
    ) -> Optional[Subscription]:
        """Cancel a subscription."""
        subscription = await self.get(db, subscription_id)
        if not subscription:
            return None
        
        subscription.status = SubscriptionStatus.CANCELED
        subscription.canceled_at = datetime.now(timezone.utc)
        db.add(subscription)
        await db.commit()
        await db.refresh(subscription)
        return subscription

    async def get_expiring_soon(
        self,
        db: AsyncSession,
        *,
        days: int = 7,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Subscription]:
        """Get subscriptions expiring soon."""
        from datetime import timedelta
        
        expiry_threshold = datetime.now(timezone.utc) + timedelta(days=days)
        
        query = (
            select(self.model)
            .where(
                and_(
                    self.model.status == SubscriptionStatus.ACTIVE,
                    self.model.current_period_end <= expiry_threshold,
                )
            )
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_subscriber_count(
        self,
        db: AsyncSession,
        *,
        creator_id: UUID,
    ) -> int:
        """Get count of active subscribers for a creator."""
        query = select(func.count()).select_from(self.model).where(
            and_(
                self.model.creator_id == creator_id,
                self.model.status == SubscriptionStatus.ACTIVE,
            )
        )
        result = await db.execute(query)
        return result.scalar_one()


class CRUDPayout(CRUDBase[Payout, BaseSchema, BaseSchema]):
    """CRUD operations for Payout model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
        status: Optional[PayoutStatus] = None,
    ) -> List[Payout]:
        """Get payouts by user."""
        query = select(self.model).where(self.model.user_id == user_id)
        
        if status:
            query = query.where(self.model.status == status)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_pending(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Payout]:
        """Get pending payouts."""
        query = (
            select(self.model)
            .where(self.model.status == PayoutStatus.PENDING)
            .order_by(self.model.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def update_status(
        self,
        db: AsyncSession,
        *,
        payout_id: UUID,
        status: PayoutStatus,
    ) -> Optional[Payout]:
        """Update payout status."""
        payout = await self.get(db, payout_id)
        if not payout:
            return None
        
        payout.status = status
        if status == PayoutStatus.PAID:
            payout.paid_at = datetime.now(timezone.utc)
        
        db.add(payout)
        await db.commit()
        await db.refresh(payout)
        return payout

    async def get_total_earnings(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> float:
        """Calculate total earnings from paid payouts."""
        query = select(func.sum(self.model.amount)).where(
            and_(
                self.model.user_id == user_id,
                self.model.status == PayoutStatus.PAID,
            )
        )
        
        if start_date:
            query = query.where(self.model.created_at >= start_date)
        if end_date:
            query = query.where(self.model.created_at <= end_date)
        
        result = await db.execute(query)
        total = result.scalar_one_or_none()
        return float(total) if total else 0.0


class CRUDTransaction(CRUDBase[Transaction, BaseSchema, BaseSchema]):
    """CRUD operations for Transaction model."""

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
        transaction_type: Optional[str] = None,
    ) -> List[Transaction]:
        """Get transactions by user."""
        query = select(self.model).where(self.model.user_id == user_id)
        
        if transaction_type:
            query = query.where(self.model.transaction_type == transaction_type)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_balance(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
    ) -> float:
        """Calculate user's current balance from all transactions."""
        # Sum credits (income)
        credit_query = select(func.sum(self.model.amount)).where(
            and_(
                self.model.user_id == user_id,
                self.model.transaction_type.in_([
                    "subscription_payment",
                    "ad_revenue",
                    "donation",
                    "tip",
                ])
            )
        )
        credit_result = await db.execute(credit_query)
        credits = credit_result.scalar_one_or_none() or 0
        
        # Sum debits (expenses)
        debit_query = select(func.sum(self.model.amount)).where(
            and_(
                self.model.user_id == user_id,
                self.model.transaction_type.in_([
                    "payout",
                    "refund",
                ])
            )
        )
        debit_result = await db.execute(debit_query)
        debits = debit_result.scalar_one_or_none() or 0
        
        return float(credits - debits)

    async def get_transaction_summary(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """Get transaction summary for a user."""
        query = select(self.model).where(self.model.user_id == user_id)
        
        if start_date:
            query = query.where(self.model.created_at >= start_date)
        if end_date:
            query = query.where(self.model.created_at <= end_date)
        
        result = await db.execute(query)
        transactions = result.scalars().all()
        
        summary = {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_balance": 0.0,
            "by_type": {},
        }
        
        income_types = [
            "subscription_payment",
            "ad_revenue",
            "donation",
            "tip",
        ]
        
        for transaction in transactions:
            amount = float(transaction.amount)
            type_name = transaction.transaction_type
            
            if transaction.transaction_type in income_types:
                summary["total_income"] += amount
            else:
                summary["total_expenses"] += amount
            
            if type_name not in summary["by_type"]:
                summary["by_type"][type_name] = 0.0
            summary["by_type"][type_name] += amount
        
        summary["net_balance"] = summary["total_income"] - summary["total_expenses"]
        
        return summary


# Create singleton instances
payment = CRUDPayment(Payment)
subscription = CRUDSubscription(Subscription)
payout = CRUDPayout(Payout)
transaction = CRUDTransaction(Transaction)
