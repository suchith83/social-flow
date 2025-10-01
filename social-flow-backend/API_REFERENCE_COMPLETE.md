# Social Flow API Reference - Complete Guide

**Version**: 1.0.0  
**Base URL**: `https://api.socialflow.com/api/v1`  
**Date**: October 2025

---

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Error Handling](#error-handling)
4. [API Endpoints](#api-endpoints)
   - [Authentication & Users](#authentication--users)
   - [Posts & Feed](#posts--feed)
   - [Videos](#videos)
   - [Live Streaming](#live-streaming)
   - [Comments](#comments)
   - [Notifications](#notifications)
   - [Payments & Subscriptions](#payments--subscriptions)
   - [Ads & Monetization](#ads--monetization)
   - [AI/ML Features](#aiml-features)
   - [Analytics](#analytics)
   - [Health & Monitoring](#health--monitoring)
5. [Webhooks](#webhooks)
6. [WebSocket Events](#websocket-events)

---

## Authentication

### Authentication Methods

Social Flow API supports three authentication methods:

#### 1. JWT Bearer Token (Recommended)
```http
Authorization: Bearer <access_token>
```

#### 2. API Key
```http
X-API-Key: <your_api_key>
```

#### 3. OAuth 2.0
Supported providers: Google, Facebook, GitHub

### Getting Access Tokens

**Register User**
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecureP@ss123",
  "full_name": "John Doe"
}

Response 201:
{
  "id": "uuid",
  "email": "user@example.com",
  "username": "johndoe",
  "created_at": "2025-10-01T12:00:00Z"
}
```

**Login**
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=johndoe&password=SecureP@ss123

Response 200:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Refresh Token**
```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}

Response 200:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Token Expiration
- **Access Token**: 1 hour
- **Refresh Token**: 7 days
- **API Key**: Never expires (revocable)

---

## Rate Limiting

### Rate Limit Tiers

| Tier | Requests/Minute | Requests/Hour | Requests/Day |
|------|----------------|---------------|--------------|
| Free | 60 | 1,000 | 10,000 |
| Basic | 120 | 5,000 | 100,000 |
| Pro | 300 | 20,000 | 500,000 |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Rate Limit Headers

Every API response includes rate limit information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1633104000
```

### Rate Limit Exceeded Response

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Try again in 30 seconds.",
  "retry_after": 30
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "Specific field error"
  },
  "request_id": "req_abc123def456",
  "timestamp": "2025-10-01T12:00:00Z"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request successful, no content to return |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Authenticated but not authorized |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource already exists |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

- `invalid_request`: Request validation failed
- `authentication_required`: Missing authentication
- `invalid_token`: Token expired or invalid
- `permission_denied`: Insufficient permissions
- `resource_not_found`: Requested resource doesn't exist
- `resource_exists`: Resource already exists (duplicate)
- `rate_limit_exceeded`: Too many requests
- `validation_error`: Input validation failed
- `internal_error`: Server-side error

---

## API Endpoints

### Authentication & Users

#### POST /auth/register
Create a new user account.

**Request:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecureP@ss123",
  "full_name": "John Doe",
  "date_of_birth": "1990-01-01"
}
```

**Response 201:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_verified": false,
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### POST /auth/login
Authenticate user and receive tokens.

**Request:**
```http
Content-Type: application/x-www-form-urlencoded

username=johndoe&password=SecureP@ss123
```

**Response 200:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "johndoe",
    "email": "user@example.com"
  }
}
```

---

#### POST /auth/2fa/enable
Enable two-factor authentication.

**Request:**
```json
{
  "method": "totp"
}
```

**Response 200:**
```json
{
  "secret": "JBSWY3DPEHPK3PXP",
  "qr_code": "data:image/png;base64,iVBORw0KG...",
  "backup_codes": [
    "12345678",
    "87654321"
  ]
}
```

---

#### POST /auth/2fa/verify
Verify 2FA code during login.

**Request:**
```json
{
  "code": "123456"
}
```

**Response 200:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

---

#### GET /users/me
Get current user profile.

**Response 200:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "johndoe",
  "email": "user@example.com",
  "full_name": "John Doe",
  "bio": "Content creator",
  "avatar_url": "https://cdn.socialflow.com/avatars/johndoe.jpg",
  "followers_count": 1234,
  "following_count": 567,
  "is_verified": true,
  "is_premium": true,
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

#### PUT /users/me
Update current user profile.

**Request:**
```json
{
  "full_name": "John Smith",
  "bio": "Professional content creator",
  "website": "https://johnsmith.com"
}
```

**Response 200:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "johndoe",
  "full_name": "John Smith",
  "bio": "Professional content creator",
  "website": "https://johnsmith.com",
  "updated_at": "2025-10-01T12:00:00Z"
}
```

---

#### POST /users/{user_id}/follow
Follow a user.

**Response 204:** No Content

---

#### DELETE /users/{user_id}/follow
Unfollow a user.

**Response 204:** No Content

---

#### GET /users/{user_id}/followers
Get user's followers.

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `limit` (integer): Items per page (default: 20, max: 100)

**Response 200:**
```json
{
  "data": [
    {
      "id": "uuid",
      "username": "follower1",
      "avatar_url": "https://cdn.socialflow.com/avatars/follower1.jpg",
      "is_verified": false
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1234,
    "has_next": true
  }
}
```

---

### Posts & Feed

#### POST /posts
Create a new post.

**Request:**
```json
{
  "content": "Hello, Social Flow! #firstpost",
  "media_urls": [
    "https://cdn.socialflow.com/media/image1.jpg"
  ],
  "visibility": "public",
  "mentions": ["@johndoe"],
  "location": "San Francisco, CA"
}
```

**Response 201:**
```json
{
  "id": "uuid",
  "content": "Hello, Social Flow! #firstpost",
  "media_urls": ["https://cdn.socialflow.com/media/image1.jpg"],
  "author": {
    "id": "uuid",
    "username": "johndoe",
    "avatar_url": "https://cdn.socialflow.com/avatars/johndoe.jpg"
  },
  "likes_count": 0,
  "comments_count": 0,
  "reposts_count": 0,
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### GET /posts/{post_id}
Get a specific post.

**Response 200:**
```json
{
  "id": "uuid",
  "content": "Hello, Social Flow!",
  "author": {
    "id": "uuid",
    "username": "johndoe",
    "avatar_url": "https://cdn.socialflow.com/avatars/johndoe.jpg",
    "is_verified": true
  },
  "likes_count": 42,
  "comments_count": 10,
  "reposts_count": 5,
  "is_liked": false,
  "is_reposted": false,
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### PUT /posts/{post_id}
Update a post.

**Request:**
```json
{
  "content": "Updated content #edited"
}
```

**Response 200:**
```json
{
  "id": "uuid",
  "content": "Updated content #edited",
  "edited_at": "2025-10-01T13:00:00Z"
}
```

---

#### DELETE /posts/{post_id}
Delete a post.

**Response 204:** No Content

---

#### POST /posts/{post_id}/like
Like a post.

**Response 204:** No Content

---

#### DELETE /posts/{post_id}/like
Unlike a post.

**Response 204:** No Content

---

#### POST /posts/repost
Repost (share) a post.

**Request:**
```json
{
  "post_id": "uuid",
  "comment": "Great content!"
}
```

**Response 201:**
```json
{
  "id": "uuid",
  "original_post": {
    "id": "uuid",
    "content": "Original post content"
  },
  "comment": "Great content!",
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### GET /posts/feed
Get personalized feed (ML-ranked).

**Query Parameters:**
- `page` (integer): Page number
- `limit` (integer): Items per page (max: 50)
- `algorithm` (string): `ml_ranked` | `chronological` | `trending`

**Response 200:**
```json
{
  "data": [
    {
      "id": "uuid",
      "content": "Post content",
      "author": {},
      "engagement_score": 0.95,
      "created_at": "2025-10-01T12:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "has_next": true
  }
}
```

---

### Videos

#### POST /videos/upload
Upload a video file.

**Request:**
```http
Content-Type: multipart/form-data

file: <video_file>
title: "My Video Title"
description: "Video description"
visibility: "public"
tags: ["tech", "tutorial"]
```

**Response 201:**
```json
{
  "id": "uuid",
  "title": "My Video Title",
  "status": "processing",
  "upload_url": "https://cdn.socialflow.com/uploads/video.mp4",
  "thumbnail_url": null,
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### GET /videos/{video_id}
Get video details.

**Response 200:**
```json
{
  "id": "uuid",
  "title": "My Video Title",
  "description": "Video description",
  "status": "ready",
  "duration": 120,
  "views_count": 1000,
  "likes_count": 50,
  "author": {
    "id": "uuid",
    "username": "johndoe"
  },
  "video_url": "https://cdn.socialflow.com/videos/video.m3u8",
  "thumbnail_url": "https://cdn.socialflow.com/thumbnails/video.jpg",
  "resolutions": ["360p", "720p", "1080p"],
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### GET /videos/{video_id}/stream
Get video streaming URL with adaptive bitrate.

**Response 200:**
```json
{
  "manifest_url": "https://cdn.socialflow.com/videos/video.m3u8",
  "formats": [
    {
      "resolution": "1080p",
      "bitrate": 5000000,
      "url": "https://cdn.socialflow.com/videos/video_1080p.mp4"
    },
    {
      "resolution": "720p",
      "bitrate": 2500000,
      "url": "https://cdn.socialflow.com/videos/video_720p.mp4"
    }
  ]
}
```

---

#### POST /videos/{video_id}/like
Like a video.

**Response 204:** No Content

---

#### POST /videos/{video_id}/view
Record a video view.

**Request:**
```json
{
  "watch_time": 45,
  "completed": false
}
```

**Response 204:** No Content

---

#### POST /videos/upload/initiate
Initiate chunked upload for large videos.

**Request:**
```json
{
  "filename": "large_video.mp4",
  "filesize": 524288000,
  "chunk_size": 5242880,
  "content_type": "video/mp4"
}
```

**Response 201:**
```json
{
  "upload_id": "uuid",
  "chunk_count": 100,
  "chunk_size": 5242880,
  "upload_urls": [
    "https://cdn.socialflow.com/uploads/chunk_1",
    "https://cdn.socialflow.com/uploads/chunk_2"
  ]
}
```

---

#### PUT /videos/upload/{upload_id}/chunk/{chunk_number}
Upload a video chunk.

**Request:**
```http
Content-Type: application/octet-stream

<binary_chunk_data>
```

**Response 200:**
```json
{
  "chunk_number": 1,
  "uploaded": true,
  "progress": 1.0
}
```

---

#### POST /videos/upload/{upload_id}/complete
Complete chunked upload.

**Response 200:**
```json
{
  "video_id": "uuid",
  "status": "processing",
  "message": "Video uploaded successfully. Processing started."
}
```

---

### Live Streaming

#### POST /live-streaming/create
Create a live stream.

**Request:**
```json
{
  "title": "Live Coding Session",
  "description": "Building a web app",
  "scheduled_start": "2025-10-01T15:00:00Z",
  "visibility": "public"
}
```

**Response 201:**
```json
{
  "id": "uuid",
  "title": "Live Coding Session",
  "stream_key": "live_sk_abc123def456",
  "rtmp_url": "rtmp://live.socialflow.com/live",
  "hls_url": "https://live.socialflow.com/hls/stream.m3u8",
  "webrtc_url": "https://live.socialflow.com/webrtc/stream",
  "status": "scheduled",
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### POST /live-streaming/{stream_id}/start
Start a live stream.

**Response 200:**
```json
{
  "id": "uuid",
  "status": "live",
  "started_at": "2025-10-01T15:00:00Z",
  "viewer_count": 0
}
```

---

#### POST /live-streaming/{stream_id}/stop
Stop a live stream.

**Response 200:**
```json
{
  "id": "uuid",
  "status": "ended",
  "ended_at": "2025-10-01T16:00:00Z",
  "duration": 3600,
  "peak_viewers": 150,
  "total_views": 500
}
```

---

#### GET /live-streaming/{stream_id}
Get live stream details.

**Response 200:**
```json
{
  "id": "uuid",
  "title": "Live Coding Session",
  "status": "live",
  "viewer_count": 42,
  "hls_url": "https://live.socialflow.com/hls/stream.m3u8",
  "chat_enabled": true,
  "started_at": "2025-10-01T15:00:00Z"
}
```

---

#### GET /live-streaming
Get all live streams.

**Query Parameters:**
- `status` (string): `live` | `scheduled` | `ended`
- `page` (integer): Page number
- `limit` (integer): Items per page

**Response 200:**
```json
{
  "data": [
    {
      "id": "uuid",
      "title": "Live stream",
      "status": "live",
      "viewer_count": 100,
      "thumbnail_url": "https://cdn.socialflow.com/thumbnails/stream.jpg"
    }
  ],
  "pagination": {
    "page": 1,
    "total": 50
  }
}
```

---

#### GET /live-streaming/{stream_id}/analytics
Get live stream analytics.

**Response 200:**
```json
{
  "stream_id": "uuid",
  "total_views": 500,
  "peak_viewers": 150,
  "average_watch_time": 1200,
  "engagement_rate": 0.85,
  "chat_messages": 350,
  "viewer_retention": [
    {"time": 0, "viewers": 100},
    {"time": 60, "viewers": 120}
  ]
}
```

---

### Comments

#### POST /posts/{post_id}/comments
Add a comment to a post.

**Request:**
```json
{
  "content": "Great post!",
  "parent_id": null
}
```

**Response 201:**
```json
{
  "id": "uuid",
  "content": "Great post!",
  "author": {
    "id": "uuid",
    "username": "johndoe"
  },
  "likes_count": 0,
  "replies_count": 0,
  "created_at": "2025-10-01T12:00:00Z"
}
```

---

#### GET /posts/{post_id}/comments
Get comments for a post.

**Query Parameters:**
- `page` (integer): Page number
- `limit` (integer): Items per page
- `sort` (string): `recent` | `popular`

**Response 200:**
```json
{
  "data": [
    {
      "id": "uuid",
      "content": "Great post!",
      "author": {},
      "likes_count": 5,
      "created_at": "2025-10-01T12:00:00Z"
    }
  ],
  "pagination": {}
}
```

---

### Notifications

#### GET /notifications
Get user notifications.

**Query Parameters:**
- `type` (string): `like` | `comment` | `follow` | `mention`
- `read` (boolean): Filter by read status
- `page` (integer): Page number

**Response 200:**
```json
{
  "data": [
    {
      "id": "uuid",
      "type": "like",
      "message": "johndoe liked your post",
      "actor": {
        "id": "uuid",
        "username": "johndoe"
      },
      "target": {
        "type": "post",
        "id": "uuid"
      },
      "is_read": false,
      "created_at": "2025-10-01T12:00:00Z"
    }
  ],
  "unread_count": 5
}
```

---

#### POST /notifications/{notification_id}/read
Mark notification as read.

**Response 204:** No Content

---

#### POST /notifications/read-all
Mark all notifications as read.

**Response 204:** No Content

---

#### GET /notifications/preferences
Get notification preferences.

**Response 200:**
```json
{
  "email_notifications": true,
  "push_notifications": true,
  "likes": true,
  "comments": true,
  "follows": true,
  "mentions": true,
  "live_streams": false
}
```

---

#### PUT /notifications/preferences
Update notification preferences.

**Request:**
```json
{
  "email_notifications": false,
  "push_notifications": true
}
```

**Response 200:**
```json
{
  "email_notifications": false,
  "push_notifications": true,
  "updated_at": "2025-10-01T12:00:00Z"
}
```

---

### Payments & Subscriptions

#### POST /payments/stripe/payment-intent
Create a payment intent.

**Request:**
```json
{
  "amount": 999,
  "currency": "usd",
  "description": "Premium subscription"
}
```

**Response 201:**
```json
{
  "client_secret": "pi_abc123_secret_def456",
  "payment_intent_id": "pi_abc123",
  "amount": 999,
  "currency": "usd"
}
```

---

#### POST /payments/stripe/subscriptions/create
Create a subscription.

**Request:**
```json
{
  "price_id": "price_abc123",
  "payment_method_id": "pm_abc123"
}
```

**Response 201:**
```json
{
  "subscription_id": "sub_abc123",
  "status": "active",
  "current_period_end": "2025-11-01T00:00:00Z",
  "cancel_at_period_end": false
}
```

---

#### GET /payments/stripe/subscriptions
Get user subscriptions.

**Response 200:**
```json
{
  "data": [
    {
      "id": "sub_abc123",
      "status": "active",
      "plan": "Premium",
      "amount": 999,
      "currency": "usd",
      "current_period_end": "2025-11-01T00:00:00Z"
    }
  ]
}
```

---

#### DELETE /payments/stripe/subscriptions/{subscription_id}/cancel
Cancel a subscription.

**Response 200:**
```json
{
  "subscription_id": "sub_abc123",
  "status": "canceled",
  "canceled_at": "2025-10-01T12:00:00Z"
}
```

---

#### POST /payments/stripe/connect/account
Create connected account for creator payouts.

**Request:**
```json
{
  "type": "express",
  "country": "US",
  "email": "creator@example.com"
}
```

**Response 201:**
```json
{
  "account_id": "acct_abc123",
  "onboarding_url": "https://connect.stripe.com/setup/..."
}
```

---

#### POST /payments/stripe/connect/payout
Create payout to creator.

**Request:**
```json
{
  "amount": 10000,
  "currency": "usd",
  "destination": "acct_abc123"
}
```

**Response 201:**
```json
{
  "payout_id": "po_abc123",
  "amount": 10000,
  "status": "pending",
  "arrival_date": "2025-10-03T00:00:00Z"
}
```

---

### AI/ML Features

#### POST /ml/analyze
Analyze content with AI.

**Request:**
```json
{
  "content": "This is a post about AI",
  "analysis_types": ["sentiment", "topics", "toxicity"]
}
```

**Response 200:**
```json
{
  "sentiment": {
    "score": 0.85,
    "label": "positive"
  },
  "topics": ["technology", "artificial-intelligence"],
  "toxicity": {
    "score": 0.05,
    "is_toxic": false
  },
  "language": "en"
}
```

---

#### POST /ml/moderate
Moderate content.

**Request:**
```json
{
  "content_id": "uuid",
  "content_type": "post",
  "rules": ["hate_speech", "violence", "spam"]
}
```

**Response 200:**
```json
{
  "content_id": "uuid",
  "is_approved": false,
  "violations": ["hate_speech"],
  "confidence": 0.92,
  "action": "remove"
}
```

---

#### GET /ml/recommendations
Get AI-powered content recommendations.

**Query Parameters:**
- `limit` (integer): Number of recommendations
- `type` (string): `posts` | `videos` | `users`

**Response 200:**
```json
{
  "recommendations": [
    {
      "id": "uuid",
      "type": "post",
      "score": 0.95,
      "reason": "based_on_interests"
    }
  ]
}
```

---

#### POST /ml/predict-viral
Predict if content will go viral.

**Request:**
```json
{
  "content_id": "uuid",
  "content_type": "video"
}
```

**Response 200:**
```json
{
  "viral_score": 0.78,
  "predicted_views": 50000,
  "predicted_engagement": 0.12,
  "factors": [
    "trending_topic",
    "high_quality",
    "optimal_length"
  ]
}
```

---

### Analytics

#### GET /analytics/user/{user_id}
Get user analytics.

**Query Parameters:**
- `period` (string): `day` | `week` | `month` | `year`
- `start_date` (string): ISO 8601 date
- `end_date` (string): ISO 8601 date

**Response 200:**
```json
{
  "user_id": "uuid",
  "period": "month",
  "metrics": {
    "total_views": 50000,
    "total_likes": 2500,
    "total_comments": 1200,
    "followers_gained": 150,
    "engagement_rate": 0.08
  },
  "top_posts": [
    {
      "id": "uuid",
      "views": 10000,
      "engagement_rate": 0.15
    }
  ]
}
```

---

#### GET /analytics/revenue
Get revenue analytics.

**Response 200:**
```json
{
  "total_revenue": 125000,
  "revenue_by_source": {
    "subscriptions": 100000,
    "donations": 15000,
    "ads": 10000
  },
  "revenue_trend": [
    {"date": "2025-09-01", "amount": 40000},
    {"date": "2025-10-01", "amount": 42500}
  ]
}
```

---

### Health & Monitoring

#### GET /health
Basic health check.

**Response 200:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-01T12:00:00Z"
}
```

---

#### GET /health/detailed
Detailed health check with dependencies.

**Response 200:**
```json
{
  "status": "healthy",
  "database": {
    "status": "healthy",
    "response_time_ms": 5
  },
  "redis": {
    "status": "healthy",
    "response_time_ms": 2
  },
  "s3": {
    "status": "healthy"
  },
  "celery": {
    "status": "healthy",
    "workers": 4
  }
}
```

---

## Webhooks

Social Flow can send webhook notifications for important events.

### Webhook Configuration

**Create Webhook**
```http
POST /webhooks
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["post.created", "video.processed", "payment.succeeded"],
  "secret": "whsec_abc123"
}
```

### Webhook Events

| Event | Description |
|-------|-------------|
| `post.created` | New post created |
| `post.updated` | Post updated |
| `post.deleted` | Post deleted |
| `video.uploaded` | Video uploaded |
| `video.processed` | Video processing completed |
| `video.failed` | Video processing failed |
| `comment.created` | New comment |
| `user.followed` | User followed |
| `payment.succeeded` | Payment successful |
| `payment.failed` | Payment failed |
| `subscription.created` | Subscription created |
| `subscription.canceled` | Subscription canceled |
| `stream.started` | Live stream started |
| `stream.ended` | Live stream ended |

### Webhook Payload

```json
{
  "id": "evt_abc123",
  "type": "post.created",
  "created": 1633104000,
  "data": {
    "object": {
      "id": "uuid",
      "type": "post",
      "content": "New post"
    }
  }
}
```

### Webhook Signature Verification

Verify webhook authenticity using HMAC-SHA256:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

---

## WebSocket Events

Real-time updates via WebSocket connection.

### Connection

```javascript
const ws = new WebSocket('wss://api.socialflow.com/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: 'your_access_token'
  }));
};
```

### Subscribe to Events

```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['notifications', 'live_chat']
}));
```

### Event Types

**New Notification**
```json
{
  "type": "notification",
  "data": {
    "id": "uuid",
    "message": "johndoe liked your post"
  }
}
```

**Live Chat Message**
```json
{
  "type": "chat_message",
  "stream_id": "uuid",
  "data": {
    "user": "johndoe",
    "message": "Hello!",
    "timestamp": "2025-10-01T12:00:00Z"
  }
}
```

**Viewer Count Update**
```json
{
  "type": "viewer_count",
  "stream_id": "uuid",
  "count": 42
}
```

---

## SDK & Client Libraries

Official SDKs available for:

- **JavaScript/TypeScript**: `npm install @socialflow/js-sdk`
- **Python**: `pip install socialflow-sdk`
- **iOS**: CocoaPods / Swift Package Manager
- **Android**: Maven / Gradle
- **Flutter**: `flutter pub add socialflow_sdk`

### Example (JavaScript)

```javascript
import SocialFlow from '@socialflow/js-sdk';

const client = new SocialFlow({
  apiKey: 'your_api_key'
});

// Get user feed
const feed = await client.posts.getFeed({
  algorithm: 'ml_ranked',
  limit: 20
});

// Create post
const post = await client.posts.create({
  content: 'Hello from SDK!',
  visibility: 'public'
});
```

---

## Support & Resources

- **API Status**: https://status.socialflow.com
- **Developer Portal**: https://developers.socialflow.com
- **Community Forum**: https://community.socialflow.com
- **Support Email**: api-support@socialflow.com
- **Rate Limit Issues**: ratelimit@socialflow.com

---

**Last Updated**: October 1, 2025  
**API Version**: 1.0.0
