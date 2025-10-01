"""
Comprehensive Stripe Payment Service.

Handles all Stripe payment operations including:
- One-time payments (tips, content purchases)
- Subscription management
- Creator payouts via Stripe Connect
- Webhook handling
- Payment method management
- Refunds and disputes
"""

import stripe
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.core.config import settings
from app.core.exceptions import PaymentServiceError, ValidationError
from app.auth.models.user import User
from app.payments.models.payment import Payment, PaymentStatus, PaymentType
from app.auth.models.subscription import Subscription, SubscriptionStatus, SubscriptionTier
from app.payments.models.stripe_connect import (
    StripeConnectAccount,
    ConnectAccountStatus,
    CreatorPayout,
    PayoutStatus,
    WebhookEvent,
)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


class StripePaymentService:
    """Comprehensive Stripe payment service."""
    
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis
        self.platform_fee_percentage = 10.0  # 10% platform fee
        self.stripe_fee_percentage = 2.9  # Stripe's fee
        self.stripe_fee_fixed = 0.30  # Stripe's fixed fee
    
    # ============================================================================
    # ONE-TIME PAYMENTS (Tips, Content Purchases)
    # ============================================================================
    
    async def create_payment_intent(
        self,
        user_id: str,
        amount: float,
        currency: str = "usd",
        payment_type: str = PaymentType.ONE_TIME,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create Stripe PaymentIntent for one-time payment."""
        try:
            # Get user
            user = await self.db.get(User, uuid.UUID(user_id))
            if not user:
                raise ValidationError("User not found")
            
            # Get or create Stripe customer
            customer_id = await self._get_or_create_customer(user)
            
            # Create payment intent
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency,
                customer=customer_id,
                description=description or f"{payment_type} payment",
                metadata={
                    "user_id": user_id,
                    "payment_type": payment_type,
                    **(metadata or {}),
                },
                automatic_payment_methods={"enabled": True},
            )
            
            # Calculate fees
            stripe_fee = (amount * self.stripe_fee_percentage / 100) + self.stripe_fee_fixed
            platform_fee = amount * self.platform_fee_percentage / 100
            net_amount = amount - stripe_fee - platform_fee
            
            # Create payment record
            payment = Payment(
                user_id=uuid.UUID(user_id),
                amount=amount,
                currency=currency,
                payment_type=payment_type,
                status=PaymentStatus.PENDING,
                provider="stripe",
                provider_payment_id=intent.id,
                description=description,
                metadata=json.dumps(metadata) if metadata else None,
                processing_fee=stripe_fee,
                platform_fee=platform_fee,
                net_amount=net_amount,
            )
            
            self.db.add(payment)
            await self.db.commit()
            await self.db.refresh(payment)
            
            return {
                "payment_id": str(payment.id),
                "client_secret": intent.client_secret,
                "stripe_payment_intent_id": intent.id,
                "amount": amount,
                "currency": currency,
                "status": "requires_payment_method",
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to create payment intent: {str(e)}")
    
    async def confirm_payment(self, payment_id: str) -> Dict[str, Any]:
        """Confirm payment after successful charge."""
        try:
            # Get payment
            result = await self.db.execute(
                select(Payment).where(Payment.id == uuid.UUID(payment_id))
            )
            payment = result.scalar_one_or_none()
            
            if not payment:
                raise ValidationError("Payment not found")
            
            # Retrieve PaymentIntent from Stripe
            intent = stripe.PaymentIntent.retrieve(payment.provider_payment_id)
            
            if intent.status == "succeeded":
                payment.status = PaymentStatus.COMPLETED
                payment.processed_at = datetime.utcnow()
                payment.provider_transaction_id = intent.latest_charge
                
                # Update payment method details
                if intent.payment_method:
                    pm = stripe.PaymentMethod.retrieve(intent.payment_method)
                    payment.payment_method_id = pm.id
                    payment.payment_method_type = pm.type
                    
                    if pm.type == "card":
                        payment.card_brand = pm.card.brand
                        payment.last_four_digits = pm.card.last4
                        payment.card_exp_month = pm.card.exp_month
                        payment.card_exp_year = pm.card.exp_year
                
                await self.db.commit()
                
                return {
                    "payment_id": str(payment.id),
                    "status": "completed",
                    "amount": payment.amount,
                    "currency": payment.currency,
                }
            
            return {
                "payment_id": str(payment.id),
                "status": intent.status,
                "amount": payment.amount,
                "currency": payment.currency,
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to confirm payment: {str(e)}")
    
    async def refund_payment(
        self,
        payment_id: str,
        amount: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Refund a payment (full or partial)."""
        try:
            # Get payment
            result = await self.db.execute(
                select(Payment).where(Payment.id == uuid.UUID(payment_id))
            )
            payment = result.scalar_one_or_none()
            
            if not payment:
                raise ValidationError("Payment not found")
            
            if payment.status != PaymentStatus.COMPLETED:
                raise ValidationError("Cannot refund incomplete payment")
            
            # Default to full refund
            refund_amount = amount or (payment.amount - payment.refund_amount)
            
            if refund_amount > (payment.amount - payment.refund_amount):
                raise ValidationError("Refund amount exceeds available amount")
            
            # Create Stripe refund
            refund = stripe.Refund.create(
                payment_intent=payment.provider_payment_id,
                amount=int(refund_amount * 100),
                reason=reason or "requested_by_customer",
            )
            
            # Update payment
            payment.refund_amount += refund_amount
            payment.refund_reason = reason
            payment.refunded_at = datetime.utcnow()
            
            if payment.refund_amount >= payment.amount:
                payment.status = PaymentStatus.REFUNDED
            else:
                payment.status = PaymentStatus.PARTIALLY_REFUNDED
            
            await self.db.commit()
            
            return {
                "payment_id": str(payment.id),
                "refund_id": refund.id,
                "refund_amount": refund_amount,
                "total_refunded": payment.refund_amount,
                "status": payment.status,
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to refund payment: {str(e)}")
    
    # ============================================================================
    # SUBSCRIPTION MANAGEMENT
    # ============================================================================
    
    async def create_subscription(
        self,
        user_id: str,
        tier: str,
        payment_method_id: str,
        trial_days: int = 0,
    ) -> Dict[str, Any]:
        """Create new subscription."""
        try:
            # Get user
            user = await self.db.get(User, uuid.UUID(user_id))
            if not user:
                raise ValidationError("User not found")
            
            # Get pricing for tier
            price_data = self._get_subscription_pricing(tier)
            
            # Get or create Stripe customer
            customer_id = await self._get_or_create_customer(user)
            
            # Attach payment method
            stripe.PaymentMethod.attach(payment_method_id, customer=customer_id)
            
            # Set as default payment method
            stripe.Customer.modify(
                customer_id,
                invoice_settings={"default_payment_method": payment_method_id},
            )
            
            # Create Stripe subscription
            stripe_sub = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_data["stripe_price_id"]}],
                trial_period_days=trial_days if trial_days > 0 else None,
                metadata={"user_id": user_id, "tier": tier},
            )
            
            # Create subscription record
            start_date = datetime.utcnow()
            end_date = None  # Recurring subscription
            
            if trial_days > 0:
                trial_start = start_date
                trial_end = start_date + timedelta(days=trial_days)
            else:
                trial_start = None
                trial_end = None
            
            subscription = Subscription(
                user_id=uuid.UUID(user_id),
                tier=tier,
                status=SubscriptionStatus.TRIAL if trial_days > 0 else SubscriptionStatus.ACTIVE,
                price=price_data["price"],
                currency=price_data["currency"],
                billing_cycle=price_data["billing_cycle"],
                is_trial=trial_days > 0,
                trial_days=trial_days,
                start_date=start_date,
                end_date=end_date,
                trial_start_date=trial_start,
                trial_end_date=trial_end,
                provider="stripe",
                provider_subscription_id=stripe_sub.id,
                provider_customer_id=customer_id,
                payment_method_id=payment_method_id,
                features=json.dumps(price_data["features"]),
                limits=json.dumps(price_data["limits"]),
            )
            
            self.db.add(subscription)
            await self.db.commit()
            await self.db.refresh(subscription)
            
            return {
                "subscription_id": str(subscription.id),
                "stripe_subscription_id": stripe_sub.id,
                "tier": tier,
                "status": subscription.status,
                "price": price_data["price"],
                "billing_cycle": price_data["billing_cycle"],
                "trial_days": trial_days,
                "next_billing_date": stripe_sub.current_period_end,
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to create subscription: {str(e)}")
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
    ) -> Dict[str, Any]:
        """Cancel subscription."""
        try:
            # Get subscription
            result = await self.db.execute(
                select(Subscription).where(Subscription.id == uuid.UUID(subscription_id))
            )
            subscription = result.scalar_one_or_none()
            
            if not subscription:
                raise ValidationError("Subscription not found")
            
            # Cancel in Stripe
            stripe_sub = stripe.Subscription.retrieve(subscription.provider_subscription_id)
            
            if immediate:
                stripe.Subscription.delete(subscription.provider_subscription_id)
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.end_date = datetime.utcnow()
            else:
                # Cancel at period end
                stripe.Subscription.modify(
                    subscription.provider_subscription_id,
                    cancel_at_period_end=True,
                )
                subscription.status = SubscriptionStatus.ACTIVE  # Still active until period end
                subscription.end_date = datetime.fromtimestamp(stripe_sub.current_period_end)
            
            subscription.cancelled_at = datetime.utcnow()
            await self.db.commit()
            
            return {
                "subscription_id": str(subscription.id),
                "status": subscription.status,
                "end_date": subscription.end_date.isoformat() if subscription.end_date else None,
                "cancelled_immediately": immediate,
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to cancel subscription: {str(e)}")
    
    async def update_subscription(
        self,
        subscription_id: str,
        new_tier: str,
    ) -> Dict[str, Any]:
        """Update subscription tier (upgrade/downgrade)."""
        try:
            # Get subscription
            result = await self.db.execute(
                select(Subscription).where(Subscription.id == uuid.UUID(subscription_id))
            )
            subscription = result.scalar_one_or_none()
            
            if not subscription:
                raise ValidationError("Subscription not found")
            
            # Get new pricing
            price_data = self._get_subscription_pricing(new_tier)
            
            # Update Stripe subscription
            stripe_sub = stripe.Subscription.retrieve(subscription.provider_subscription_id)
            stripe.Subscription.modify(
                subscription.provider_subscription_id,
                items=[{
                    "id": stripe_sub["items"]["data"][0].id,
                    "price": price_data["stripe_price_id"],
                }],
                proration_behavior="create_prorations",
            )
            
            # Update subscription record
            subscription.tier = new_tier
            subscription.price = price_data["price"]
            subscription.billing_cycle = price_data["billing_cycle"]
            subscription.features = json.dumps(price_data["features"])
            subscription.limits = json.dumps(price_data["limits"])
            
            await self.db.commit()
            
            return {
                "subscription_id": str(subscription.id),
                "tier": new_tier,
                "price": price_data["price"],
                "billing_cycle": price_data["billing_cycle"],
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to update subscription: {str(e)}")
    
    # ============================================================================
    # STRIPE CONNECT - CREATOR PAYOUTS
    # ============================================================================
    
    async def create_connect_account(
        self,
        user_id: str,
        country: str = "US",
        account_type: str = "express",
    ) -> Dict[str, Any]:
        """Create Stripe Connect account for creator."""
        try:
            # Get user
            user = await self.db.get(User, uuid.UUID(user_id))
            if not user:
                raise ValidationError("User not found")
            
            # Check if account already exists
            result = await self.db.execute(
                select(StripeConnectAccount).where(StripeConnectAccount.user_id == user.id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                raise ValidationError("Connect account already exists for this user")
            
            # Create Stripe Connect account
            account = stripe.Account.create(
                type=account_type,
                country=country,
                email=user.email,
                capabilities={
                    "card_payments": {"requested": True},
                    "transfers": {"requested": True},
                },
                business_profile={
                    "url": user.website or f"https://socialflow.com/@{user.username}",
                },
                metadata={"user_id": user_id, "username": user.username},
            )
            
            # Create account link for onboarding
            account_link = stripe.AccountLink.create(
                account=account.id,
                refresh_url=f"{settings.FRONTEND_URL}/settings/payouts/refresh",
                return_url=f"{settings.FRONTEND_URL}/settings/payouts/complete",
                type="account_onboarding",
            )
            
            # Create database record
            connect_account = StripeConnectAccount(
                user_id=user.id,
                stripe_account_id=account.id,
                account_type=account_type,
                country=country,
                status=ConnectAccountStatus.PENDING,
                onboarding_url=account_link.url,
            )
            
            self.db.add(connect_account)
            await self.db.commit()
            await self.db.refresh(connect_account)
            
            return {
                "connect_account_id": str(connect_account.id),
                "stripe_account_id": account.id,
                "onboarding_url": account_link.url,
                "status": connect_account.status,
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to create connect account: {str(e)}")
    
    async def get_connect_account_status(self, user_id: str) -> Dict[str, Any]:
        """Get Connect account status and requirements."""
        try:
            # Get connect account
            result = await self.db.execute(
                select(StripeConnectAccount).where(
                    StripeConnectAccount.user_id == uuid.UUID(user_id)
                )
            )
            connect_account = result.scalar_one_or_none()
            
            if not connect_account:
                raise ValidationError("Connect account not found")
            
            # Get latest status from Stripe
            account = stripe.Account.retrieve(connect_account.stripe_account_id)
            
            # Update database record
            connect_account.charges_enabled = account.charges_enabled
            connect_account.payouts_enabled = account.payouts_enabled
            connect_account.details_submitted = account.details_submitted
            
            if account.requirements:
                connect_account.requirements_currently_due = json.dumps(
                    account.requirements.currently_due
                )
                connect_account.requirements_eventually_due = json.dumps(
                    account.requirements.eventually_due
                )
                connect_account.requirements_past_due = json.dumps(
                    account.requirements.past_due
                )
            
            if connect_account.is_fully_onboarded:
                connect_account.status = ConnectAccountStatus.ACTIVE
                if not connect_account.onboarding_completed_at:
                    connect_account.onboarding_completed_at = datetime.utcnow()
            
            await self.db.commit()
            
            return {
                "connect_account_id": str(connect_account.id),
                "stripe_account_id": connect_account.stripe_account_id,
                "status": connect_account.status,
                "charges_enabled": connect_account.charges_enabled,
                "payouts_enabled": connect_account.payouts_enabled,
                "details_submitted": connect_account.details_submitted,
                "is_fully_onboarded": connect_account.is_fully_onboarded,
                "requirements_currently_due": json.loads(
                    connect_account.requirements_currently_due or "[]"
                ),
                "available_balance": connect_account.available_balance,
                "pending_balance": connect_account.pending_balance,
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to get connect account status: {str(e)}")
    
    async def create_payout(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime,
        revenue_breakdown: Dict[str, float],
    ) -> Dict[str, Any]:
        """Create creator payout."""
        try:
            # Get connect account
            result = await self.db.execute(
                select(StripeConnectAccount).where(
                    StripeConnectAccount.user_id == uuid.UUID(user_id)
                )
            )
            connect_account = result.scalar_one_or_none()
            
            if not connect_account:
                raise ValidationError("Connect account not found")
            
            if not connect_account.is_fully_onboarded:
                raise ValidationError("Connect account is not fully onboarded")
            
            # Calculate total revenue
            gross_amount = sum(revenue_breakdown.values())
            
            if gross_amount <= 0:
                raise ValidationError("No revenue to payout")
            
            # Calculate fees
            platform_fee = gross_amount * (self.platform_fee_percentage / 100)
            stripe_fee = (gross_amount * (self.stripe_fee_percentage / 100)) + self.stripe_fee_fixed
            net_amount = gross_amount - platform_fee - stripe_fee
            
            # Minimum payout threshold
            if net_amount < 10.0:
                raise ValidationError("Payout amount below minimum threshold ($10)")
            
            # Create payout record
            payout = CreatorPayout(
                user_id=uuid.UUID(user_id),
                connect_account_id=connect_account.id,
                amount=net_amount,
                currency=connect_account.currency,
                status=PayoutStatus.PENDING,
                gross_amount=gross_amount,
                platform_fee=platform_fee,
                stripe_fee=stripe_fee,
                net_amount=net_amount,
                subscription_revenue=revenue_breakdown.get("subscription", 0.0),
                tips_revenue=revenue_breakdown.get("tips", 0.0),
                content_sales_revenue=revenue_breakdown.get("content_sales", 0.0),
                ad_revenue=revenue_breakdown.get("ad_revenue", 0.0),
                period_start=period_start,
                period_end=period_end,
                description=f"Creator payout for {period_start.date()} to {period_end.date()}",
            )
            
            self.db.add(payout)
            await self.db.commit()
            await self.db.refresh(payout)
            
            # Create Stripe transfer
            transfer = stripe.Transfer.create(
                amount=int(net_amount * 100),
                currency=connect_account.currency.lower(),
                destination=connect_account.stripe_account_id,
                description=payout.description,
                metadata={
                    "payout_id": str(payout.id),
                    "user_id": user_id,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                },
            )
            
            # Update payout with Stripe IDs
            payout.stripe_transfer_id = transfer.id
            payout.status = PayoutStatus.PROCESSING
            payout.processed_at = datetime.utcnow()
            
            await self.db.commit()
            
            return {
                "payout_id": str(payout.id),
                "stripe_transfer_id": transfer.id,
                "amount": net_amount,
                "currency": connect_account.currency,
                "status": payout.status,
                "gross_amount": gross_amount,
                "platform_fee": platform_fee,
                "stripe_fee": stripe_fee,
            }
            
        except stripe.error.StripeError as e:
            raise PaymentServiceError(f"Stripe error: {str(e)}")
        except Exception as e:
            raise PaymentServiceError(f"Failed to create payout: {str(e)}")
    
    # ============================================================================
    # WEBHOOK HANDLING
    # ============================================================================
    
    async def handle_webhook(
        self,
        payload: bytes,
        signature: str,
    ) -> Dict[str, Any]:
        """Handle Stripe webhook event."""
        try:
            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload, signature, settings.STRIPE_WEBHOOK_SECRET
            )
            
            # Store webhook event
            webhook_event = WebhookEvent(
                stripe_event_id=event.id,
                event_type=event.type,
                event_version=event.api_version,
                event_data=json.dumps(event.data),
            )
            
            self.db.add(webhook_event)
            await self.db.commit()
            
            # Process event based on type
            if event.type == "payment_intent.succeeded":
                await self._handle_payment_intent_succeeded(event.data.object)
            elif event.type == "payment_intent.payment_failed":
                await self._handle_payment_intent_failed(event.data.object)
            elif event.type == "invoice.payment_succeeded":
                await self._handle_invoice_payment_succeeded(event.data.object)
            elif event.type == "invoice.payment_failed":
                await self._handle_invoice_payment_failed(event.data.object)
            elif event.type == "customer.subscription.deleted":
                await self._handle_subscription_deleted(event.data.object)
            elif event.type == "account.updated":
                await self._handle_account_updated(event.data.object)
            elif event.type == "payout.paid":
                await self._handle_payout_paid(event.data.object)
            elif event.type == "payout.failed":
                await self._handle_payout_failed(event.data.object)
            
            # Mark webhook as processed
            webhook_event.is_processed = True
            webhook_event.processed_at = datetime.utcnow()
            await self.db.commit()
            
            return {"status": "success", "event_type": event.type}
            
        except stripe.error.SignatureVerificationError:
            raise PaymentServiceError("Invalid webhook signature")
        except Exception as e:
            # Log error and mark webhook as failed
            if 'webhook_event' in locals():
                webhook_event.processing_error = str(e)
                webhook_event.retry_count += 1
                await self.db.commit()
            raise PaymentServiceError(f"Failed to process webhook: {str(e)}")
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    async def _get_or_create_customer(self, user: User) -> str:
        """Get or create Stripe customer for user."""
        # Check cache first
        cache_key = f"stripe_customer:{user.id}"
        cached_id = await self.redis.get(cache_key)
        
        if cached_id:
            return cached_id.decode()
        
        # Search for existing customer by email
        customers = stripe.Customer.list(email=user.email, limit=1)
        
        if customers.data:
            customer_id = customers.data[0].id
        else:
            # Create new customer
            customer = stripe.Customer.create(
                email=user.email,
                name=user.display_name,
                metadata={"user_id": str(user.id), "username": user.username},
            )
            customer_id = customer.id
        
        # Cache customer ID
        await self.redis.setex(cache_key, 86400, customer_id)  # 24 hours
        
        return customer_id
    
    def _get_subscription_pricing(self, tier: str) -> Dict[str, Any]:
        """Get subscription pricing details."""
        pricing = {
            SubscriptionTier.FREE: {
                "price": 0.0,
                "currency": "USD",
                "billing_cycle": "lifetime",
                "stripe_price_id": None,
                "features": ["basic_features", "limited_storage"],
                "limits": {"videos": 10, "posts": 50, "storage_gb": 5},
            },
            SubscriptionTier.BASIC: {
                "price": 9.99,
                "currency": "USD",
                "billing_cycle": "monthly",
                "stripe_price_id": settings.STRIPE_BASIC_PRICE_ID,
                "features": ["ad_free", "hd_streaming", "basic_analytics"],
                "limits": {"videos": 100, "posts": 500, "storage_gb": 50},
            },
            SubscriptionTier.PREMIUM: {
                "price": 19.99,
                "currency": "USD",
                "billing_cycle": "monthly",
                "stripe_price_id": settings.STRIPE_PREMIUM_PRICE_ID,
                "features": [
                    "all_basic_features",
                    "4k_streaming",
                    "advanced_analytics",
                    "live_streaming",
                    "monetization",
                ],
                "limits": {"videos": -1, "posts": -1, "storage_gb": 200},
            },
            SubscriptionTier.PRO: {
                "price": 49.99,
                "currency": "USD",
                "billing_cycle": "monthly",
                "stripe_price_id": settings.STRIPE_PRO_PRICE_ID,
                "features": [
                    "all_premium_features",
                    "priority_support",
                    "custom_branding",
                    "api_access",
                    "white_label",
                ],
                "limits": {"videos": -1, "posts": -1, "storage_gb": 500},
            },
        }
        
        return pricing.get(tier, pricing[SubscriptionTier.FREE])
    
    async def _handle_payment_intent_succeeded(self, payment_intent):
        """Handle successful payment intent."""
        # Update payment record
        result = await self.db.execute(
            select(Payment).where(Payment.provider_payment_id == payment_intent.id)
        )
        payment = result.scalar_one_or_none()
        
        if payment:
            payment.status = PaymentStatus.COMPLETED
            payment.processed_at = datetime.utcnow()
            await self.db.commit()
    
    async def _handle_payment_intent_failed(self, payment_intent):
        """Handle failed payment intent."""
        result = await self.db.execute(
            select(Payment).where(Payment.provider_payment_id == payment_intent.id)
        )
        payment = result.scalar_one_or_none()
        
        if payment:
            payment.status = PaymentStatus.FAILED
            payment.failed_at = datetime.utcnow()
            await self.db.commit()
    
    async def _handle_invoice_payment_succeeded(self, invoice):
        """Handle successful subscription invoice payment."""
        # Update subscription status
        result = await self.db.execute(
            select(Subscription).where(
                Subscription.provider_subscription_id == invoice.subscription
            )
        )
        subscription = result.scalar_one_or_none()
        
        if subscription:
            subscription.status = SubscriptionStatus.ACTIVE
            await self.db.commit()
    
    async def _handle_invoice_payment_failed(self, invoice):
        """Handle failed subscription invoice payment."""
        result = await self.db.execute(
            select(Subscription).where(
                Subscription.provider_subscription_id == invoice.subscription
            )
        )
        subscription = result.scalar_one_or_none()
        
        if subscription:
            subscription.status = SubscriptionStatus.INACTIVE
            await self.db.commit()
    
    async def _handle_subscription_deleted(self, subscription):
        """Handle subscription deletion."""
        result = await self.db.execute(
            select(Subscription).where(
                Subscription.provider_subscription_id == subscription.id
            )
        )
        db_subscription = result.scalar_one_or_none()
        
        if db_subscription:
            db_subscription.status = SubscriptionStatus.CANCELLED
            db_subscription.end_date = datetime.utcnow()
            await self.db.commit()
    
    async def _handle_account_updated(self, account):
        """Handle Connect account update."""
        result = await self.db.execute(
            select(StripeConnectAccount).where(
                StripeConnectAccount.stripe_account_id == account.id
            )
        )
        connect_account = result.scalar_one_or_none()
        
        if connect_account:
            connect_account.charges_enabled = account.charges_enabled
            connect_account.payouts_enabled = account.payouts_enabled
            connect_account.details_submitted = account.details_submitted
            
            if connect_account.is_fully_onboarded:
                connect_account.status = ConnectAccountStatus.ACTIVE
            
            await self.db.commit()
    
    async def _handle_payout_paid(self, payout):
        """Handle successful payout."""
        # Find payout by Stripe transfer ID
        result = await self.db.execute(
            select(CreatorPayout).where(
                CreatorPayout.stripe_payout_id == payout.id
            )
        )
        creator_payout = result.scalar_one_or_none()
        
        if creator_payout:
            creator_payout.status = PayoutStatus.PAID
            creator_payout.paid_at = datetime.utcnow()
            await self.db.commit()
    
    async def _handle_payout_failed(self, payout):
        """Handle failed payout."""
        result = await self.db.execute(
            select(CreatorPayout).where(
                CreatorPayout.stripe_payout_id == payout.id
            )
        )
        creator_payout = result.scalar_one_or_none()
        
        if creator_payout:
            creator_payout.status = PayoutStatus.FAILED
            creator_payout.failed_at = datetime.utcnow()
            creator_payout.failure_message = payout.failure_message
            await self.db.commit()
