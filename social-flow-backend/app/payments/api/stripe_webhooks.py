"""
Stripe Webhook API Routes.

Handle Stripe webhook events for payment processing.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.payments.services.stripe_payment_service import StripePaymentService


router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@router.post(
    "/stripe",
    status_code=status.HTTP_200_OK,
    summary="Stripe Webhook Handler",
    description="Handle Stripe webhook events (payment, subscription, account updates)."
)
async def handle_stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Stripe webhook events.
    
    This endpoint receives real-time events from Stripe for:
    - Payment intent updates (succeeded, failed)
    - Invoice payment updates
    - Subscription lifecycle events
    - Connect account updates
    - Payout updates
    
    Args:
        request: FastAPI request with raw body
        stripe_signature: Stripe webhook signature header
        db: Database session
        
    Returns:
        Success response for Stripe
    """
    if not stripe_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header"
        )
    
    # Get raw request body
    try:
        payload = await request.body()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read request body: {str(e)}"
        )
    
    # Process webhook
    service = StripePaymentService(db)
    
    try:
        await service.handle_webhook(
            payload=payload,
            signature=stripe_signature
        )
        
        return {"status": "success", "message": "Webhook processed"}
        
    except ValueError as e:
        # Invalid signature or malformed payload
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Log error but return 200 to prevent Stripe retries for unrecoverable errors
        # Stripe will retry webhooks that return 4xx/5xx errors
        print(f"Webhook processing error: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.get(
    "/stripe/test",
    status_code=status.HTTP_200_OK,
    summary="Test Webhook Endpoint",
    description="Test endpoint to verify webhook URL is accessible."
)
async def test_webhook():
    """
    Test webhook endpoint accessibility.
    
    This endpoint can be used to verify that the webhook URL is accessible
    from the internet before configuring it in Stripe dashboard.
    
    Returns:
        Success message
    """
    return {
        "status": "success",
        "message": "Webhook endpoint is accessible",
        "endpoint": "/api/v1/webhooks/stripe"
    }
