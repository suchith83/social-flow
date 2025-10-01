# Webhooks Implementation Guide

## Overview

Webhooks allow you to receive real-time notifications about events in your Social Flow account. Instead of polling the API, your application can subscribe to specific events and receive HTTP POST requests whenever those events occur.

## Table of Contents

1. [Webhook Basics](#webhook-basics)
2. [Event Types](#event-types)
3. [Webhook Registration](#webhook-registration)
4. [Payload Format](#payload-format)
5. [Security & Verification](#security--verification)
6. [Retry Logic](#retry-logic)
7. [Testing Webhooks](#testing-webhooks)
8. [Best Practices](#best-practices)
9. [Code Examples](#code-examples)
10. [Troubleshooting](#troubleshooting)

## Webhook Basics

### What are Webhooks?

Webhooks are HTTP callbacks that deliver event data to your server in real-time. When an event occurs (e.g., a new post is created), Social Flow sends an HTTP POST request to your configured webhook URL.

### Key Benefits

- **Real-time Updates:** Receive events as they happen
- **Efficient:** No need to poll the API repeatedly
- **Scalable:** Handle thousands of events without API rate limits
- **Reliable:** Automatic retries for failed deliveries

### Architecture

```text
[Social Flow Event] → [Webhook System] → [HTTP POST] → [Your Server]
                                                ↓
                                        [Signature Verification]
                                                ↓
                                        [Event Processing]
```

## Event Types

### User Events

| Event | Description | Trigger |
|-------|-------------|---------|
| `user.created` | New user registered | User completes registration |
| `user.updated` | User profile updated | User changes profile info |
| `user.deleted` | User account deleted | User deletes their account |
| `user.verified` | User verified email | User verifies email address |
| `user.banned` | User account banned | Admin bans user |
| `user.unbanned` | User account unbanned | Admin unbans user |

### Post Events

| Event | Description | Trigger |
|-------|-------------|---------|
| `post.created` | New post published | User creates a post |
| `post.updated` | Post content updated | User edits their post |
| `post.deleted` | Post removed | User or admin deletes post |
| `post.liked` | Post received a like | User likes a post |
| `post.commented` | New comment added | User comments on post |

### Video Events

| Event | Description | Trigger |
|-------|-------------|---------|
| `video.uploaded` | Video upload started | User initiates upload |
| `video.processing` | Video processing started | Transcoding begins |
| `video.processed` | Video ready for streaming | Transcoding completes |
| `video.failed` | Video processing failed | Transcoding error |
| `video.deleted` | Video removed | User deletes video |
| `video.view_milestone` | View count milestone | Video reaches 1K, 10K, 100K, 1M views |

### Live Streaming Events

| Event | Description | Trigger |
|-------|-------------|---------|
| `stream.created` | Live stream created | User creates stream |
| `stream.started` | Stream went live | Broadcaster starts streaming |
| `stream.ended` | Stream finished | Broadcaster stops streaming |
| `stream.viewer_joined` | New viewer joined | User joins stream |
| `stream.viewer_left` | Viewer left | User leaves stream |

### Payment Events

| Event | Description | Trigger |
|-------|-------------|---------|
| `payment.succeeded` | Payment successful | Successful charge |
| `payment.failed` | Payment failed | Charge declined |
| `payment.refunded` | Payment refunded | Refund processed |
| `subscription.created` | New subscription | User subscribes |
| `subscription.updated` | Subscription changed | Plan upgrade/downgrade |
| `subscription.canceled` | Subscription ended | User cancels subscription |
| `payout.initiated` | Payout started | Creator payout initiated |
| `payout.completed` | Payout successful | Payout transferred |
| `payout.failed` | Payout failed | Payout error |

### Moderation Events

| Event | Description | Trigger |
|-------|-------------|---------|
| `content.flagged` | Content flagged for review | User reports content |
| `content.approved` | Content approved | Moderator approves |
| `content.rejected` | Content rejected | Moderator rejects |
| `content.auto_moderated` | AI moderation action | AI detects violation |

## Webhook Registration

### Create a Webhook

**Endpoint:** `POST /api/v1/webhooks`

**Request:**

```json
{
  "url": "https://your-domain.com/webhooks/socialflow",
  "events": [
    "post.created",
    "video.processed",
    "payment.succeeded"
  ],
  "description": "Production webhook",
  "secret": "your-secret-key-for-signature-verification",
  "active": true
}
```

**Response:**

```json
{
  "id": "wh_1a2b3c4d",
  "url": "https://your-domain.com/webhooks/socialflow",
  "events": [
    "post.created",
    "video.processed",
    "payment.succeeded"
  ],
  "description": "Production webhook",
  "secret": "whs_xxxxxxxxxxxxx",
  "active": true,
  "created_at": "2025-01-15T10:30:00Z",
  "status": "active"
}
```

### List Webhooks

**Endpoint:** `GET /api/v1/webhooks`

**Response:**

```json
{
  "data": [
    {
      "id": "wh_1a2b3c4d",
      "url": "https://your-domain.com/webhooks/socialflow",
      "events": ["post.created", "video.processed"],
      "active": true,
      "created_at": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

### Update a Webhook

**Endpoint:** `PUT /api/v1/webhooks/{webhook_id}`

**Request:**

```json
{
  "events": [
    "post.created",
    "post.updated",
    "video.processed"
  ],
  "active": true
}
```

### Delete a Webhook

**Endpoint:** `DELETE /api/v1/webhooks/{webhook_id}`

**Response:** `204 No Content`

### Test a Webhook

**Endpoint:** `POST /api/v1/webhooks/{webhook_id}/test`

Sends a test event to verify your webhook endpoint is working.

**Response:**

```json
{
  "status": "success",
  "response_code": 200,
  "response_time_ms": 145,
  "message": "Test webhook delivered successfully"
}
```

## Payload Format

### Standard Webhook Payload

All webhook events follow this format:

```json
{
  "id": "evt_1a2b3c4d5e6f",
  "type": "post.created",
  "created": 1705318200,
  "data": {
    "object": {
      "id": "post_abc123",
      "user_id": "user_xyz789",
      "content": "Hello, Social Flow!",
      "created_at": "2025-01-15T10:30:00Z",
      "likes_count": 0,
      "comments_count": 0
    }
  },
  "account_id": "acc_123456",
  "api_version": "v1"
}
```

### Field Descriptions

- **id:** Unique event identifier
- **type:** Event type (e.g., `post.created`)
- **created:** Unix timestamp when event occurred
- **data.object:** The resource object related to the event
- **account_id:** Your Social Flow account ID
- **api_version:** API version that triggered the event

### Example Payloads

#### Post Created

```json
{
  "id": "evt_post_created_001",
  "type": "post.created",
  "created": 1705318200,
  "data": {
    "object": {
      "id": "post_abc123",
      "user_id": "user_xyz789",
      "content": "Just launched my new video!",
      "visibility": "public",
      "media_urls": ["https://cdn.socialflow.com/media/image1.jpg"],
      "created_at": "2025-01-15T10:30:00Z"
    }
  },
  "account_id": "acc_123456",
  "api_version": "v1"
}
```

#### Video Processed

```json
{
  "id": "evt_video_proc_001",
  "type": "video.processed",
  "created": 1705318800,
  "data": {
    "object": {
      "id": "video_def456",
      "user_id": "user_xyz789",
      "title": "My Tutorial Video",
      "duration": 300,
      "status": "processed",
      "hls_url": "https://cdn.socialflow.com/videos/video_def456/master.m3u8",
      "thumbnail_url": "https://cdn.socialflow.com/thumbnails/video_def456.jpg",
      "processing_time_seconds": 180,
      "created_at": "2025-01-15T10:35:00Z"
    }
  },
  "account_id": "acc_123456",
  "api_version": "v1"
}
```

#### Payment Succeeded

```json
{
  "id": "evt_payment_succ_001",
  "type": "payment.succeeded",
  "created": 1705319400,
  "data": {
    "object": {
      "id": "pay_ghi789",
      "user_id": "user_xyz789",
      "amount": 999,
      "currency": "usd",
      "status": "succeeded",
      "description": "Premium subscription",
      "payment_method": "card",
      "created_at": "2025-01-15T10:40:00Z"
    }
  },
  "account_id": "acc_123456",
  "api_version": "v1"
}
```

## Security & Verification

### Why Verify Webhooks?

Webhook verification ensures that the requests are actually from Social Flow and not from malicious actors.

### HMAC Signature Verification

Social Flow signs each webhook with HMAC-SHA256 and includes the signature in the `X-Signature` header.

### Signature Header Format

```text
X-Signature: t=1705318200,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
```

- **t:** Timestamp when the webhook was sent
- **v1:** HMAC-SHA256 signature

### Verification Steps

1. Extract the timestamp and signature from the header
2. Concatenate the timestamp and raw request body: `{timestamp}.{body}`
3. Compute HMAC-SHA256 using your webhook secret
4. Compare computed signature with received signature
5. Verify timestamp is within 5 minutes (prevent replay attacks)

### Python Example

```python
import hmac
import hashlib
import time

def verify_webhook_signature(payload_body, signature_header, webhook_secret):
    """
    Verify webhook signature from Social Flow.
    
    Args:
        payload_body: Raw request body (bytes)
        signature_header: Value of X-Signature header
        webhook_secret: Your webhook secret
    
    Returns:
        bool: True if signature is valid
    """
    try:
        # Parse signature header
        parts = dict(item.split('=') for item in signature_header.split(','))
        timestamp = parts['t']
        received_signature = parts['v1']
        
        # Check timestamp (within 5 minutes)
        current_time = int(time.time())
        if abs(current_time - int(timestamp)) > 300:
            return False
        
        # Compute expected signature
        signed_payload = f"{timestamp}.{payload_body.decode('utf-8')}"
        expected_signature = hmac.new(
            webhook_secret.encode('utf-8'),
            signed_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures (constant-time comparison)
        return hmac.compare_digest(expected_signature, received_signature)
    
    except Exception as e:
        print(f"Signature verification failed: {e}")
        return False
```

### Node.js Example

```javascript
const crypto = require('crypto');

function verifyWebhookSignature(payloadBody, signatureHeader, webhookSecret) {
  try {
    // Parse signature header
    const parts = {};
    signatureHeader.split(',').forEach(part => {
      const [key, value] = part.split('=');
      parts[key] = value;
    });
    
    const timestamp = parts.t;
    const receivedSignature = parts.v1;
    
    // Check timestamp (within 5 minutes)
    const currentTime = Math.floor(Date.now() / 1000);
    if (Math.abs(currentTime - parseInt(timestamp)) > 300) {
      return false;
    }
    
    // Compute expected signature
    const signedPayload = `${timestamp}.${payloadBody}`;
    const expectedSignature = crypto
      .createHmac('sha256', webhookSecret)
      .update(signedPayload)
      .digest('hex');
    
    // Compare signatures (constant-time comparison)
    return crypto.timingSafeEqual(
      Buffer.from(expectedSignature),
      Buffer.from(receivedSignature)
    );
  } catch (error) {
    console.error('Signature verification failed:', error);
    return false;
  }
}
```

### Java Example

```java
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Map;

public class WebhookVerifier {
    public static boolean verifySignature(String payloadBody, 
                                          String signatureHeader, 
                                          String webhookSecret) {
        try {
            // Parse signature header
            Map<String, String> parts = new HashMap<>();
            for (String part : signatureHeader.split(",")) {
                String[] keyValue = part.split("=");
                parts.put(keyValue[0], keyValue[1]);
            }
            
            String timestamp = parts.get("t");
            String receivedSignature = parts.get("v1");
            
            // Check timestamp (within 5 minutes)
            long currentTime = System.currentTimeMillis() / 1000;
            if (Math.abs(currentTime - Long.parseLong(timestamp)) > 300) {
                return false;
            }
            
            // Compute expected signature
            String signedPayload = timestamp + "." + payloadBody;
            Mac hmac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secretKey = new SecretKeySpec(
                webhookSecret.getBytes(StandardCharsets.UTF_8), 
                "HmacSHA256"
            );
            hmac.init(secretKey);
            byte[] hash = hmac.doFinal(signedPayload.getBytes(StandardCharsets.UTF_8));
            
            String expectedSignature = bytesToHex(hash);
            
            // Compare signatures
            return MessageDigest.isEqual(
                expectedSignature.getBytes(StandardCharsets.UTF_8),
                receivedSignature.getBytes(StandardCharsets.UTF_8)
            );
        } catch (Exception e) {
            System.err.println("Signature verification failed: " + e.getMessage());
            return false;
        }
    }
    
    private static String bytesToHex(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte b : bytes) {
            result.append(String.format("%02x", b));
        }
        return result.toString();
    }
}
```

## Retry Logic

### Delivery Guarantee

Social Flow attempts to deliver each webhook with exponential backoff:

| Attempt | Delay | Total Time Elapsed |
|---------|-------|-------------------|
| 1 | 0 seconds | 0 |
| 2 | 1 minute | 1 minute |
| 3 | 5 minutes | 6 minutes |
| 4 | 15 minutes | 21 minutes |
| 5 | 1 hour | 1 hour 21 minutes |
| 6 | 4 hours | 5 hours 21 minutes |
| 7 | 12 hours | 17 hours 21 minutes |

### Success Criteria

A webhook delivery is considered successful if:
- HTTP response code is 2xx (200-299)
- Response received within 10 seconds

### Failure Conditions

A webhook delivery fails if:
- HTTP response code is 4xx or 5xx
- Request times out (>10 seconds)
- Network error occurs
- DNS lookup fails

### Webhook Status

You can check the delivery status:

**Endpoint:** `GET /api/v1/webhooks/{webhook_id}/events`

```json
{
  "data": [
    {
      "id": "evt_001",
      "type": "post.created",
      "delivered": true,
      "attempts": 1,
      "last_attempt_at": "2025-01-15T10:30:05Z",
      "response_code": 200,
      "response_time_ms": 145
    },
    {
      "id": "evt_002",
      "type": "video.processed",
      "delivered": false,
      "attempts": 3,
      "last_attempt_at": "2025-01-15T10:41:00Z",
      "response_code": 500,
      "next_attempt_at": "2025-01-15T10:56:00Z"
    }
  ]
}
```

### Manual Retry

You can manually retry failed webhook deliveries:

**Endpoint:** `POST /api/v1/webhooks/{webhook_id}/events/{event_id}/retry`

## Testing Webhooks

### Local Development with ngrok

[ngrok](https://ngrok.com) creates a secure tunnel to your local server:

1. **Install ngrok:**
   ```bash
   # macOS
   brew install ngrok
   
   # Windows
   choco install ngrok
   
   # Linux
   wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
   unzip ngrok-stable-linux-amd64.zip
   ```

2. **Start your local server:**
   ```bash
   python app.py  # Running on http://localhost:5000
   ```

3. **Create ngrok tunnel:**
   ```bash
   ngrok http 5000
   ```

4. **Use ngrok URL for webhook:**
   ```text
   https://abc123.ngrok.io/webhooks/socialflow
   ```

### Test Events

Send test events to verify your webhook implementation:

```bash
curl -X POST https://api.socialflow.com/api/v1/webhooks/wh_123/test \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"event_type": "post.created"}'
```

### Sample Webhook Server (Python/Flask)

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = "your_webhook_secret"

@app.route('/webhooks/socialflow', methods=['POST'])
def handle_webhook():
    # Get raw body and signature
    payload_body = request.get_data()
    signature_header = request.headers.get('X-Signature')
    
    # Verify signature
    if not verify_signature(payload_body, signature_header, WEBHOOK_SECRET):
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Parse event
    event = request.get_json()
    event_type = event['type']
    
    # Process event
    if event_type == 'post.created':
        handle_post_created(event['data']['object'])
    elif event_type == 'video.processed':
        handle_video_processed(event['data']['object'])
    elif event_type == 'payment.succeeded':
        handle_payment_succeeded(event['data']['object'])
    
    # Return success
    return jsonify({'received': True}), 200

def verify_signature(payload_body, signature_header, secret):
    # Implementation from previous section
    pass

def handle_post_created(post):
    print(f"New post created: {post['id']}")
    # Your logic here

def handle_video_processed(video):
    print(f"Video processed: {video['id']}")
    # Send notification to user, etc.

def handle_payment_succeeded(payment):
    print(f"Payment succeeded: {payment['id']}")
    # Update user subscription, etc.

if __name__ == '__main__':
    app.run(port=5000)
```

## Best Practices

### 1. Return 200 Quickly

Process webhooks asynchronously:

```python
@app.route('/webhooks/socialflow', methods=['POST'])
def handle_webhook():
    # Verify signature
    if not verify_signature(...):
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Queue for async processing
    event = request.get_json()
    queue.enqueue('process_webhook', event)
    
    # Return immediately
    return jsonify({'received': True}), 200

# Process in background worker
def process_webhook(event):
    # Heavy processing here
    pass
```

### 2. Handle Duplicate Events

Use event IDs to track processed events:

```python
processed_events = set()

def handle_webhook(event):
    event_id = event['id']
    
    # Check if already processed
    if event_id in processed_events:
        return {'status': 'already_processed'}
    
    # Process event
    process_event(event)
    
    # Mark as processed
    processed_events.add(event_id)
```

### 3. Use a Queue System

Process webhooks with a message queue (Celery, RabbitMQ, Redis):

```python
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def process_webhook_task(event):
    # Process webhook asynchronously
    pass

@app.route('/webhooks/socialflow', methods=['POST'])
def handle_webhook():
    event = request.get_json()
    process_webhook_task.delay(event)
    return jsonify({'received': True}), 200
```

### 4. Log Everything

Log all webhook events for debugging:

```python
import logging

logger = logging.getLogger(__name__)

@app.route('/webhooks/socialflow', methods=['POST'])
def handle_webhook():
    event = request.get_json()
    
    logger.info(f"Received webhook: {event['type']} - {event['id']}")
    
    try:
        process_event(event)
        logger.info(f"Successfully processed: {event['id']}")
    except Exception as e:
        logger.error(f"Failed to process {event['id']}: {str(e)}")
        raise
    
    return jsonify({'received': True}), 200
```

### 5. Monitor Failures

Set up alerts for webhook failures:

```python
def handle_webhook(event):
    try:
        process_event(event)
    except Exception as e:
        # Alert on failure
        send_alert(f"Webhook processing failed: {event['id']}")
        raise
```

### 6. Use HTTPS

Always use HTTPS for webhook URLs to prevent man-in-the-middle attacks.

### 7. Disable Inactive Webhooks

Disable webhooks that consistently fail to avoid unnecessary retries.

## Troubleshooting

### Webhook Not Receiving Events

**Possible Causes:**
- Webhook is disabled
- URL is incorrect or unreachable
- Firewall blocking incoming requests
- SSL certificate issues

**Solutions:**
1. Check webhook status: `GET /api/v1/webhooks/{id}`
2. Test with ngrok for local development
3. Verify URL is publicly accessible
4. Check firewall/security group settings
5. Ensure valid SSL certificate

### Signature Verification Failing

**Possible Causes:**
- Using wrong webhook secret
- Modifying request body before verification
- Timestamp validation too strict

**Solutions:**
1. Verify you're using the correct secret from webhook creation
2. Verify signature before parsing JSON
3. Check timestamp tolerance (5 minutes default)
4. Log both received and computed signatures for comparison

### Webhooks Timing Out

**Possible Causes:**
- Processing taking too long (>10 seconds)
- Database queries blocking
- External API calls blocking

**Solutions:**
1. Return 200 immediately and process asynchronously
2. Use background job queue
3. Optimize database queries
4. Implement timeout for external calls

### Duplicate Events

**Possible Causes:**
- Retries due to slow response
- Not tracking processed event IDs

**Solutions:**
1. Return 200 quickly (< 2 seconds)
2. Use event ID to detect duplicates
3. Implement idempotent processing

## Support

### Get Help

- **Documentation:** https://docs.socialflow.com/webhooks
- **Status Page:** https://status.socialflow.com
- **Support Email:** webhooks-support@socialflow.com
- **Developer Forum:** https://forum.socialflow.com/webhooks

### Report Issues

If you encounter issues with webhooks:

1. Check the [Status Page](https://status.socialflow.com)
2. Review webhook delivery logs: `GET /api/v1/webhooks/{id}/events`
3. Contact support with:
   - Webhook ID
   - Event IDs
   - Timestamp of issue
   - Error messages/logs

## Conclusion

Webhooks provide a powerful way to build real-time integrations with Social Flow. By following this guide and best practices, you can:

- Receive real-time event notifications
- Process events efficiently and reliably
- Secure your webhook endpoints
- Handle failures gracefully
- Debug issues effectively

For more information, visit our [Developer Documentation](https://docs.socialflow.com).
