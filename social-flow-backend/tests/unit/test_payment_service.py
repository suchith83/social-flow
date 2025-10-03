"""
Unit tests for payment service functionality.

This module contains unit tests for the Stripe payment service
and subscription management.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta
from app.payments.services.payments_service import PaymentsService
from app.payments.schemas.payment import PaymentCreate, SubscriptionCreate
from app.models import Payment, Subscription, User


class TestPaymentService:
    """Test cases for PaymentService."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = AsyncMock()
        db.add = Mock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def payment_service(self, mock_db):
        """Create PaymentsService instance for testing."""
        # PaymentsService doesn't take any arguments
        return PaymentsService()

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            id="user123",
            username="testuser",
            email="test@example.com",
            stripe_customer_id="cus_test123",
        )

    @pytest.fixture
    def test_payment(self, test_user):
        """Create test payment."""
        return Payment(
            id="payment123",
            user_id=test_user.id,
            amount=1000,
            currency="USD",
            status="completed",
            provider_payment_id="pi_test123",
            payment_type="one_time",
            payment_method_id="pm_test123",
        )

    @pytest.fixture
    def test_subscription(self, test_user):
        """Create test subscription."""
        return Subscription(
            id="sub123",
            user_id=test_user.id,
            stripe_subscription_id="sub_test123",
            tier="premium",
            status="active",
            price_amount=999,
            currency="USD",
            billing_cycle="monthly",
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=30),
        )

    @pytest.mark.asyncio
    async def test_create_payment_intent_success(self, payment_service, test_user):
        """Test successful payment intent creation."""
        payment_data = PaymentCreate(
            amount=1000,
            currency="USD",
            payment_method_id="pm_test123",
        )
        
        with patch('stripe.PaymentIntent.create') as mock_stripe:
            mock_stripe.return_value = {
                "id": "pi_test123",
                "client_secret": "secret_test123",
                "status": "requires_confirmation",
                "amount": 1000,
                "currency": "usd",
            }
            
            result = await payment_service.create_payment_intent(
                test_user.id,
                payment_data.amount,
                payment_data.currency,
            )
            
            assert result["id"] == "pi_test123"
            assert "client_secret" in result
            mock_stripe.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_payment_intent_invalid_amount(self, payment_service, test_user):
        """Test payment intent creation with invalid amount."""
        with pytest.raises(ValueError):
            await payment_service.create_payment_intent(
                test_user.id,
                -100,  # Negative amount
                "USD",
            )

    @pytest.mark.asyncio
    async def test_confirm_payment_success(self, payment_service, mock_db, test_user):
        """Test successful payment confirmation."""
        payment_intent_id = "pi_test123"
        
        with patch('stripe.PaymentIntent.retrieve') as mock_retrieve:
            mock_retrieve.return_value = {
                "id": payment_intent_id,
                "status": "succeeded",
                "amount": 1000,
                "currency": "usd",
                "payment_method": "pm_test123",
            }
            
            result = await payment_service.confirm_payment(payment_intent_id, test_user.id)
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_create_subscription_success(self, payment_service, mock_db, test_user):
        """Test successful subscription creation."""
        subscription_data = SubscriptionCreate(
            plan="premium",
            payment_method_id="pm_test123",
        )
        
        with patch('stripe.Subscription.create') as mock_stripe:
            mock_stripe.return_value = {
                "id": "sub_test123",
                "status": "active",
                "items": {
                    "data": [{
                        "price": {
                            "id": "price_test123",
                            "unit_amount": 999,
                            "currency": "usd",
                            "recurring": {"interval": "month"},
                        }
                    }]
                },
                "current_period_start": 1234567890,
                "current_period_end": 1237246290,
            }
            
            result = await payment_service.create_subscription(
                test_user.id,
                subscription_data.plan,
                subscription_data.payment_method_id,
            )
            
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            assert result.plan == "premium"
            assert result.status == "active"

    @pytest.mark.asyncio
    async def test_cancel_subscription_success(self, payment_service, mock_db, test_subscription):
        """Test successful subscription cancellation."""
        with patch('stripe.Subscription.modify') as mock_stripe:
            mock_stripe.return_value = {
                "id": test_subscription.stripe_subscription_id,
                "status": "canceled",
                "cancel_at_period_end": True,
            }
            
            with patch.object(payment_service, 'get_subscription_by_id', return_value=test_subscription):
                result = await payment_service.cancel_subscription(test_subscription.id)
                
                mock_db.commit.assert_called_once()
                assert result.status == "canceled"

    @pytest.mark.asyncio
    async def test_update_subscription_plan_success(self, payment_service, mock_db, test_subscription):
        """Test successful subscription plan update."""
        new_plan = "enterprise"
        
        with patch('stripe.Subscription.modify') as mock_stripe:
            mock_stripe.return_value = {
                "id": test_subscription.stripe_subscription_id,
                "status": "active",
                "items": {
                    "data": [{
                        "id": "si_test123",
                        "price": {
                            "unit_amount": 2999,
                            "currency": "usd",
                        }
                    }]
                },
            }
            
            with patch.object(payment_service, 'get_subscription_by_id', return_value=test_subscription):
                result = await payment_service.update_subscription_plan(
                    test_subscription.id,
                    new_plan,
                )
                
                mock_db.commit.assert_called_once()
                assert result.plan == new_plan

    @pytest.mark.asyncio
    async def test_process_webhook_payment_succeeded(self, payment_service, mock_db):
        """Test processing webhook for successful payment."""
        webhook_event = {
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test123",
                    "amount": 1000,
                    "currency": "usd",
                    "status": "succeeded",
                    "metadata": {
                        "user_id": "user123",
                    }
                }
            }
        }
        
        result = await payment_service.process_webhook(webhook_event)
        
        assert result["status"] == "processed"
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_process_webhook_subscription_created(self, payment_service, mock_db):
        """Test processing webhook for subscription creation."""
        webhook_event = {
            "type": "customer.subscription.created",
            "data": {
                "object": {
                    "id": "sub_test123",
                    "status": "active",
                    "customer": "cus_test123",
                    "items": {
                        "data": [{
                            "price": {
                                "unit_amount": 999,
                                "currency": "usd",
                                "recurring": {"interval": "month"},
                            }
                        }]
                    },
                    "current_period_start": 1234567890,
                    "current_period_end": 1237246290,
                }
            }
        }
        
        result = await payment_service.process_webhook(webhook_event)
        
        assert result["status"] == "processed"

    @pytest.mark.asyncio
    async def test_get_payment_history(self, payment_service, mock_db, test_user):
        """Test getting user payment history."""
        mock_payments = [
            Payment(
                id=f"payment{i}",
                user_id=test_user.id,
                amount=1000 + i * 100,
                currency="USD",
                status="completed",
            )
            for i in range(10)
        ]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_payments
        mock_db.execute.return_value = mock_result
        
        result = await payment_service.get_payment_history(test_user.id)
        
        assert len(result) == 10
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_subscription(self, payment_service, mock_db, test_user, test_subscription):
        """Test getting active subscription for user."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = test_subscription
        mock_db.execute.return_value = mock_result
        
        result = await payment_service.get_active_subscription(test_user.id)
        
        assert result.id == test_subscription.id
        assert result.status == "active"
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_refund_payment_success(self, payment_service, mock_db, test_payment):
        """Test successful payment refund."""
        with patch('stripe.Refund.create') as mock_stripe:
            mock_stripe.return_value = {
                "id": "re_test123",
                "payment_intent": test_payment.provider_payment_id,
                "amount": test_payment.amount,
                "status": "succeeded",
            }
            
            with patch.object(payment_service, 'get_payment_by_id', return_value=test_payment):
                result = await payment_service.refund_payment(test_payment.id)
                
                mock_db.commit.assert_called_once()
                assert result.status == "refunded"
                mock_stripe.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_customer_success(self, payment_service, test_user):
        """Test creating Stripe customer."""
        with patch('stripe.Customer.create') as mock_stripe:
            mock_stripe.return_value = {
                "id": "cus_test123",
                "email": test_user.email,
                "metadata": {
                    "user_id": test_user.id,
                }
            }
            
            result = await payment_service.create_stripe_customer(test_user)
            
            assert result["id"] == "cus_test123"
            mock_stripe.assert_called_once_with(
                email=test_user.email,
                metadata={"user_id": test_user.id},
            )

    @pytest.mark.asyncio
    async def test_calculate_subscription_price(self, payment_service):
        """Test subscription price calculation for different plans."""
        # Test all plan prices
        assert payment_service.get_plan_price("basic") == 499
        assert payment_service.get_plan_price("premium") == 999
        assert payment_service.get_plan_price("business") == 1999
        assert payment_service.get_plan_price("enterprise") == 2999

    @pytest.mark.asyncio
    async def test_validate_payment_amount(self, payment_service):
        """Test payment amount validation."""
        # Valid amounts
        assert payment_service.validate_amount(100) is True
        assert payment_service.validate_amount(1000) is True
        
        # Invalid amounts
        assert payment_service.validate_amount(0) is False
        assert payment_service.validate_amount(-100) is False
        assert payment_service.validate_amount(None) is False

    @pytest.mark.asyncio
    async def test_get_revenue_analytics(self, payment_service, mock_db):
        """Test getting revenue analytics."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50000  # Total revenue in cents
        mock_db.execute.return_value = mock_result
        
        result = await payment_service.get_revenue_analytics(days=30)
        
        assert result["total_revenue"] == 50000
        assert result["period_days"] == 30
        mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_apply_coupon_success(self, payment_service, test_user):
        """Test applying coupon to subscription."""
        coupon_code = "SAVE20"
        
        with patch('stripe.Coupon.retrieve') as mock_retrieve:
            mock_retrieve.return_value = {
                "id": coupon_code,
                "percent_off": 20,
                "valid": True,
            }
            
            result = await payment_service.validate_coupon(coupon_code)
            
            assert result["valid"] is True
            assert result["percent_off"] == 20

    @pytest.mark.asyncio
    async def test_apply_coupon_invalid(self, payment_service):
        """Test applying invalid coupon."""
        coupon_code = "INVALID"
        
        with patch('stripe.Coupon.retrieve') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Coupon not found")
            
            with pytest.raises(Exception):
                await payment_service.validate_coupon(coupon_code)

    @pytest.mark.asyncio
    async def test_subscription_renewal_success(self, payment_service, mock_db, test_subscription):
        """Test automatic subscription renewal."""
        with patch('stripe.Subscription.retrieve') as mock_retrieve:
            mock_retrieve.return_value = {
                "id": test_subscription.stripe_subscription_id,
                "status": "active",
                "current_period_start": int(datetime.utcnow().timestamp()),
                "current_period_end": int((datetime.utcnow() + timedelta(days=30)).timestamp()),
            }
            
            with patch.object(payment_service, 'get_subscription_by_id', return_value=test_subscription):
                result = await payment_service.renew_subscription(test_subscription.id)
                
                mock_db.commit.assert_called_once()
                assert result.status == "active"
