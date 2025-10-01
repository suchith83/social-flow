"""
Integration tests for Payment API endpoints.

This module contains integration tests for payment and subscription endpoints
including Stripe integration and webhook handling.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch
from app.models import User, Payment, Subscription


class TestPaymentEndpoints:
    """Integration tests for payment endpoints."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_create_payment_intent(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test creating a payment intent."""
        payment_data = {
            "amount": 1000,
            "currency": "USD",
        }
        
        with patch('stripe.PaymentIntent.create') as mock_stripe:
            mock_stripe.return_value = {
                "id": "pi_test123",
                "client_secret": "secret_test123",
                "status": "requires_confirmation",
            }
            
            response = await async_client.post(
                "/api/v1/payments/create-intent",
                json=payment_data,
                headers=auth_headers,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert "client_secret" in data

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_create_payment_intent_unauthorized(self, async_client: AsyncClient):
        """Test creating payment intent without authentication."""
        payment_data = {
            "amount": 1000,
            "currency": "USD",
        }
        
        response = await async_client.post(
            "/api/v1/payments/create-intent",
            json=payment_data,
        )
        
        assert response.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_create_subscription(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test creating a subscription."""
        subscription_data = {
            "plan": "premium",
            "payment_method_id": "pm_test123",
        }
        
        with patch('stripe.Subscription.create') as mock_stripe:
            mock_stripe.return_value = {
                "id": "sub_test123",
                "status": "active",
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
            
            response = await async_client.post(
                "/api/v1/subscriptions/",
                json=subscription_data,
                headers=auth_headers,
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["plan"] == "premium"
            assert data["status"] == "active"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_get_subscription_plans(self, async_client: AsyncClient):
        """Test getting available subscription plans."""
        response = await async_client.get("/api/v1/subscriptions/plans")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Check that plan details are included
        for plan in data:
            assert "name" in plan
            assert "price" in plan

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_get_user_subscription(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test getting user's active subscription."""
        response = await async_client.get(
            "/api/v1/subscriptions/me",
            headers=auth_headers,
        )
        
        # Should return 200 with subscription or 404 if no active subscription
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_cancel_subscription(self, async_client: AsyncClient, test_subscription: Subscription, auth_headers: dict):
        """Test canceling a subscription."""
        with patch('stripe.Subscription.modify') as mock_stripe:
            mock_stripe.return_value = {
                "id": test_subscription.stripe_subscription_id,
                "status": "canceled",
                "cancel_at_period_end": True,
            }
            
            response = await async_client.delete(
                f"/api/v1/subscriptions/{test_subscription.id}",
                headers=auth_headers,
            )
            
            assert response.status_code in [200, 204]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_update_subscription_plan(self, async_client: AsyncClient, test_subscription: Subscription, auth_headers: dict):
        """Test updating subscription plan."""
        update_data = {
            "plan": "enterprise",
        }
        
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
            
            response = await async_client.patch(
                f"/api/v1/subscriptions/{test_subscription.id}",
                json=update_data,
                headers=auth_headers,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["plan"] == "enterprise"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_get_payment_history(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test getting payment history."""
        response = await async_client.get(
            "/api/v1/payments/history",
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_get_payment_by_id(self, async_client: AsyncClient, test_payment: Payment, test_user: User, auth_headers: dict):
        """Test getting a specific payment."""
        response = await async_client.get(
            f"/api/v1/payments/{test_payment.id}",
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_payment.id
        assert data["amount"] == test_payment.amount

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_refund_payment(self, async_client: AsyncClient, test_payment: Payment, auth_headers: dict):
        """Test requesting a payment refund."""
        with patch('stripe.Refund.create') as mock_stripe:
            mock_stripe.return_value = {
                "id": "re_test123",
                "payment_intent": test_payment.stripe_payment_intent_id,
                "amount": test_payment.amount,
                "status": "succeeded",
            }
            
            response = await async_client.post(
                f"/api/v1/payments/{test_payment.id}/refund",
                headers=auth_headers,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "refunded"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_stripe_webhook_payment_succeeded(self, async_client: AsyncClient):
        """Test Stripe webhook for successful payment."""
        webhook_payload = {
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
        
        # Note: Webhook signature verification would need to be mocked
        with patch('stripe.Webhook.construct_event') as mock_verify:
            mock_verify.return_value = webhook_payload
            
            response = await async_client.post(
                "/api/v1/webhooks/stripe",
                json=webhook_payload,
                headers={"Stripe-Signature": "test_signature"},
            )
            
            assert response.status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_stripe_webhook_subscription_created(self, async_client: AsyncClient):
        """Test Stripe webhook for subscription creation."""
        webhook_payload = {
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
                }
            }
        }
        
        with patch('stripe.Webhook.construct_event') as mock_verify:
            mock_verify.return_value = webhook_payload
            
            response = await async_client.post(
                "/api/v1/webhooks/stripe",
                json=webhook_payload,
                headers={"Stripe-Signature": "test_signature"},
            )
            
            assert response.status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_add_payment_method(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test adding a payment method."""
        payment_method_data = {
            "payment_method_id": "pm_test123",
        }
        
        with patch('stripe.PaymentMethod.attach') as mock_stripe:
            mock_stripe.return_value = {
                "id": "pm_test123",
                "customer": test_user.stripe_customer_id,
            }
            
            response = await async_client.post(
                "/api/v1/payments/payment-methods",
                json=payment_method_data,
                headers=auth_headers,
            )
            
            assert response.status_code in [200, 201]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_list_payment_methods(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test listing user's payment methods."""
        with patch('stripe.PaymentMethod.list') as mock_stripe:
            mock_stripe.return_value = {
                "data": [
                    {"id": "pm_test1", "type": "card"},
                    {"id": "pm_test2", "type": "card"},
                ]
            }
            
            response = await async_client.get(
                "/api/v1/payments/payment-methods",
                headers=auth_headers,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_remove_payment_method(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test removing a payment method."""
        payment_method_id = "pm_test123"
        
        with patch('stripe.PaymentMethod.detach') as mock_stripe:
            mock_stripe.return_value = {"id": payment_method_id}
            
            response = await async_client.delete(
                f"/api/v1/payments/payment-methods/{payment_method_id}",
                headers=auth_headers,
            )
            
            assert response.status_code in [200, 204]

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_get_invoice_list(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test getting list of invoices."""
        with patch('stripe.Invoice.list') as mock_stripe:
            mock_stripe.return_value = {
                "data": [
                    {
                        "id": "in_test1",
                        "amount_paid": 999,
                        "currency": "usd",
                        "status": "paid",
                    }
                ]
            }
            
            response = await async_client.get(
                "/api/v1/payments/invoices",
                headers=auth_headers,
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_download_invoice(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test downloading an invoice."""
        invoice_id = "in_test123"
        
        with patch('stripe.Invoice.retrieve') as mock_stripe:
            mock_stripe.return_value = {
                "id": invoice_id,
                "invoice_pdf": "https://example.com/invoice.pdf",
            }
            
            response = await async_client.get(
                f"/api/v1/payments/invoices/{invoice_id}/download",
                headers=auth_headers,
            )
            
            assert response.status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_apply_coupon_to_subscription(self, async_client: AsyncClient, test_subscription: Subscription, auth_headers: dict):
        """Test applying a coupon to subscription."""
        coupon_data = {
            "coupon_code": "SAVE20",
        }
        
        with patch('stripe.Coupon.retrieve') as mock_retrieve:
            mock_retrieve.return_value = {
                "id": "SAVE20",
                "percent_off": 20,
                "valid": True,
            }
            
            with patch('stripe.Subscription.modify') as mock_modify:
                mock_modify.return_value = {
                    "id": test_subscription.stripe_subscription_id,
                    "discount": {"coupon": {"id": "SAVE20", "percent_off": 20}},
                }
                
                response = await async_client.post(
                    f"/api/v1/subscriptions/{test_subscription.id}/coupon",
                    json=coupon_data,
                    headers=auth_headers,
                )
                
                assert response.status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_payment_validation_invalid_amount(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test payment creation with invalid amount."""
        payment_data = {
            "amount": -100,  # Negative amount
            "currency": "USD",
        }
        
        response = await async_client.post(
            "/api/v1/payments/create-intent",
            json=payment_data,
            headers=auth_headers,
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.payment
    async def test_subscription_plan_validation(self, async_client: AsyncClient, test_user: User, auth_headers: dict):
        """Test subscription creation with invalid plan."""
        subscription_data = {
            "plan": "invalid_plan",
            "payment_method_id": "pm_test123",
        }
        
        response = await async_client.post(
            "/api/v1/subscriptions/",
            json=subscription_data,
            headers=auth_headers,
        )
        
        assert response.status_code in [400, 422]  # Validation error
