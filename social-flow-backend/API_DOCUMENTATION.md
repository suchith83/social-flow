# API Documentation

This document provides comprehensive API documentation for the Social Flow Backend, including all endpoints, request/response formats, authentication, and examples.

## 📋 Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)
  - [Authentication](#authentication-endpoints)
  - [Users](#user-endpoints)
  - [Videos](#video-endpoints)
  - [Posts](#post-endpoints)
  - [Comments](#comment-endpoints)
  - [Likes](#like-endpoints)
  - [Follows](#follow-endpoints)
  - [Advertisements](#advertisement-endpoints)
  - [Payments](#payment-endpoints)
  - [Subscriptions](#subscription-endpoints)
  - [Notifications](#notification-endpoints)
  - [Analytics](#analytics-endpoints)
  - [Search](#search-endpoints)
  - [Admin](#admin-endpoints)
  - [Moderation](#moderation-endpoints)
  - [ML/AI](#ml-ai-endpoints)
- [WebSocket Events](#websocket-events)
- [SDK Examples](#sdk-examples)

## 🌐 Base URL

- **Development**: `http://localhost:8000/api/v1`
- **Staging**: `https://api-staging.socialflow.com/api/v1`
- **Production**: `https://api.socialflow.com/api/v1`

## 🔐 Authentication

The API uses JWT (JSON Web Token) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your_access_token>
```

### Token Types

- **Access Token**: Short-lived (30 minutes), used for API requests
- **Refresh Token**: Long-lived (7 days), used to get new access tokens

### Getting Tokens

```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

## ⚠️ Error Handling

The API uses standard HTTP status codes and returns error details in JSON format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    }
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource already exists |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## 🚦 Rate Limiting

Rate limits are applied per endpoint and user:

- **Authentication**: 5 requests per minute
- **General API**: 1000 requests per hour
- **File Upload**: 10 requests per hour
- **Search**: 100 requests per hour

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## 🔗 Endpoints

### Authentication Endpoints

#### Register User

```http
POST /auth/register
Content-Type: application/json

{
  "username": "nirmalmina",
  "email": "nirmalmina@socialflow.com",
  "password": "securepassword123",
  "display_name": "Nirmal Mina",
  "bio": "Software developer",
  "avatar_url": "https://example.com/avatar.jpg",
  "website": "https://nirmalmina.com",
  "location": "New York, NY"
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "username": "nirmalmina",
  "email": "nirmalmina@socialflow.com",
  "display_name": "Nirmal Mina",
  "bio": "Software developer",
  "avatar_url": "https://example.com/avatar.jpg",
  "website": "https://nirmalmina.com",
  "location": "New York, NY",
  "is_active": true,
  "is_verified": false,
  "followers_count": 0,
  "following_count": 0,
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### Login User

```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=nirmalmina&password=securepassword123
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

#### Refresh Token

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

### User Endpoints

#### Get Current User

```http
GET /users/me
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "username": "nirmalmina",
  "email": "nirmalmina@socialflow.com",
  "display_name": "Nirmal Mina",
  "bio": "Software developer",
  "avatar_url": "https://example.com/avatar.jpg",
  "website": "https://nirmalmina.com",
  "location": "New York, NY",
  "is_active": true,
  "is_verified": false,
  "followers_count": 150,
  "following_count": 75,
  "posts_count": 25,
  "videos_count": 10,
  "total_views": 50000,
  "total_likes": 2500,
  "privacy_level": "public",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-15T10:30:00Z",
  "last_login_at": "2025-01-15T10:30:00Z"
}
```

#### Get User by ID

```http
GET /users/{user_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "username": "nirmalmina",
  "display_name": "Nirmal Mina",
  "bio": "Software developer",
  "avatar_url": "https://example.com/avatar.jpg",
  "website": "https://nirmalmina.com",
  "location": "New York, NY",
  "is_verified": false,
  "followers_count": 150,
  "following_count": 75,
  "posts_count": 25,
  "videos_count": 10,
  "total_views": 50000,
  "total_likes": 2500,
  "privacy_level": "public",
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### Update User Profile

```http
PUT /users/me
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "display_name": "John Smith",
  "bio": "Full-stack developer",
  "website": "https://johnsmith.com",
  "location": "San Francisco, CA",
  "privacy_level": "public"
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "username": "nirmalmina",
  "email": "nirmalmina@socialflow.com",
  "display_name": "John Smith",
  "bio": "Full-stack developer",
  "avatar_url": "https://example.com/avatar.jpg",
  "website": "https://johnsmith.com",
  "location": "San Francisco, CA",
  "privacy_level": "public",
  "updated_at": "2025-01-15T10:30:00Z"
}
```

### Video Endpoints

#### Upload Video

```http
POST /videos/upload
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

file: <video_file>
title: "My Amazing Video"
description: "This is a description of my video"
tags: "funny,comedy,entertainment"
```

**Response:**
```json
{
  "video_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "uploaded",
  "message": "Video uploaded successfully",
  "upload_url": "https://s3.amazonaws.com/bucket/video.mp4",
  "processing_job_id": "job_123456789"
}
```

#### Get Video

```http
GET /videos/{video_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "My Amazing Video",
  "description": "This is a description of my video",
  "tags": "funny,comedy,entertainment",
  "filename": "video.mp4",
  "file_size": 10485760,
  "duration": 120.5,
  "resolution": "1920x1080",
  "bitrate": 5000,
  "codec": "h264",
  "thumbnail_url": "https://cloudfront.net/thumbnails/video_thumb.jpg",
  "hls_url": "https://cloudfront.net/videos/video.m3u8",
  "dash_url": "https://cloudfront.net/videos/video.mpd",
  "streaming_url": "https://cloudfront.net/videos/video.mp4",
  "status": "processed",
  "visibility": "public",
  "is_approved": true,
  "is_flagged": false,
  "views_count": 1500,
  "likes_count": 75,
  "dislikes_count": 2,
  "comments_count": 25,
  "shares_count": 10,
  "total_watch_time": 18000,
  "average_watch_time": 12.0,
  "retention_rate": 0.85,
  "is_monetized": true,
  "ad_revenue": 25.50,
  "engagement_rate": 0.12,
  "like_ratio": 0.97,
  "owner_id": "123e4567-e89b-12d3-a456-426614174000",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:05:00Z"
}
```

#### Get Video Stream URL

```http
GET /videos/{video_id}/stream?quality=720p
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "video_id": "123e4567-e89b-12d3-a456-426614174000",
  "streaming_url": "https://cloudfront.net/videos/video_720p.mp4",
  "quality": "720p",
  "duration": 120.5,
  "resolution": "1280x720",
  "bitrate": 3000,
  "expires_at": "2025-01-01T01:00:00Z"
}
```

#### Like Video

```http
POST /videos/{video_id}/like
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "liked": true,
  "likes_count": 76,
  "message": "Video liked successfully"
}
```

#### Unlike Video

```http
DELETE /videos/{video_id}/like
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "liked": false,
  "likes_count": 75,
  "message": "Video unliked successfully"
}
```

#### Record View

```http
POST /videos/{video_id}/view
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "views_count": 1501,
  "message": "View recorded successfully"
}
```

#### Create Live Stream

```http
POST /videos/live/create
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

title: "My Live Stream"
description: "Live streaming session"
```

**Response:**
```json
{
  "stream_id": "123e4567-e89b-12d3-a456-426614174000",
  "stream_key": "live_abc123def456",
  "rtmp_url": "rtmp://live.socialflow.com/live",
  "playback_url": "https://live.socialflow.com/stream/123e4567-e89b-12d3-a456-426614174000",
  "status": "active",
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### End Live Stream

```http
POST /videos/live/{stream_id}/end
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "stream_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "ended",
  "duration": 3600,
  "viewers_count": 150,
  "ended_at": "2025-01-01T01:00:00Z"
}
```

#### Get Videos List

```http
GET /videos?skip=0&limit=20&category=entertainment
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "videos": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "My Amazing Video",
      "description": "This is a description of my video",
      "thumbnail_url": "https://cloudfront.net/thumbnails/video_thumb.jpg",
      "duration": 120.5,
      "views_count": 1500,
      "likes_count": 75,
      "owner": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "username": "nirmalmina",
        "display_name": "Nirmal Mina",
        "avatar_url": "https://example.com/avatar.jpg"
      },
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 100,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

### Post Endpoints

#### Create Post

```http
POST /posts
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "content": "This is my first post! #excited #newuser",
  "media_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "visibility": "public"
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "content": "This is my first post! #excited #newuser",
  "media_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "visibility": "public",
  "likes_count": 0,
  "reposts_count": 0,
  "replies_count": 0,
  "shares_count": 0,
  "is_pinned": false,
  "is_verified": false,
  "author_id": "123e4567-e89b-12d3-a456-426614174000",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### Get Post

```http
GET /posts/{post_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "content": "This is my first post! #excited #newuser",
  "media_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "visibility": "public",
  "likes_count": 25,
  "reposts_count": 5,
  "replies_count": 3,
  "shares_count": 2,
  "is_pinned": false,
  "is_verified": false,
  "author": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "username": "nirmalmina",
    "display_name": "Nirmal Mina",
    "avatar_url": "https://example.com/avatar.jpg",
    "is_verified": false
  },
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:00:00Z"
}
```

#### Get Feed

```http
GET /posts/feed?skip=0&limit=20
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "posts": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "content": "This is my first post! #excited #newuser",
      "media_urls": [
        "https://example.com/image1.jpg"
      ],
      "likes_count": 25,
      "reposts_count": 5,
      "replies_count": 3,
      "author": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "username": "nirmalmina",
        "display_name": "Nirmal Mina",
        "avatar_url": "https://example.com/avatar.jpg"
      },
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 100,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

### Comment Endpoints

#### Create Comment

```http
POST /comments
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "content": "Great post!",
  "post_id": "123e4567-e89b-12d3-a456-426614174000",
  "parent_id": null
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "content": "Great post!",
  "post_id": "123e4567-e89b-12d3-a456-426614174000",
  "parent_id": null,
  "likes_count": 0,
  "replies_count": 0,
  "author_id": "123e4567-e89b-12d3-a456-426614174000",
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### Get Comments

```http
GET /comments?post_id=123e4567-e89b-12d3-a456-426614174000&skip=0&limit=20
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "comments": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "content": "Great post!",
      "post_id": "123e4567-e89b-12d3-a456-426614174000",
      "parent_id": null,
      "likes_count": 5,
      "replies_count": 2,
      "author": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "username": "nirmalmina",
        "display_name": "Nirmal Mina",
        "avatar_url": "https://example.com/avatar.jpg"
      },
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 50,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

### Like Endpoints

#### Like Post

```http
POST /likes
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "target_type": "post",
  "target_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response:**
```json
{
  "liked": true,
  "likes_count": 26,
  "message": "Post liked successfully"
}
```

#### Unlike Post

```http
DELETE /likes?target_type=post&target_id=123e4567-e89b-12d3-a456-426614174000
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "liked": false,
  "likes_count": 25,
  "message": "Post unliked successfully"
}
```

### Follow Endpoints

#### Follow User

```http
POST /follows
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "target_user_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response:**
```json
{
  "following": true,
  "followers_count": 151,
  "following_count": 76,
  "message": "User followed successfully"
}
```

#### Unfollow User

```http
DELETE /follows?target_user_id=123e4567-e89b-12d3-a456-426614174000
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "following": false,
  "followers_count": 150,
  "following_count": 75,
  "message": "User unfollowed successfully"
}
```

#### Get Followers

```http
GET /follows/followers?user_id=123e4567-e89b-12d3-a456-426614174000&skip=0&limit=20
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "followers": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "username": "nirmalmina",
      "display_name": "Nirmal Mina",
      "avatar_url": "https://example.com/avatar.jpg",
      "is_verified": false,
      "followed_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 150,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

#### Get Following

```http
GET /follows/following?user_id=123e4567-e89b-12d3-a456-426614174000&skip=0&limit=20
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "following": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "username": "nirmalmina",
      "display_name": "Nirmal Mina",
      "avatar_url": "https://example.com/avatar.jpg",
      "is_verified": false,
      "followed_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 75,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

### Advertisement Endpoints

#### Get Ads

```http
GET /ads?skip=0&limit=20&category=entertainment
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "ads": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Amazing Product",
      "description": "Check out this amazing product!",
      "image_url": "https://example.com/ad.jpg",
      "video_url": "https://example.com/ad.mp4",
      "click_url": "https://example.com/product",
      "category": "entertainment",
      "targeting": {
        "age_range": [18, 35],
        "interests": ["technology", "gaming"],
        "location": "US"
      },
      "budget": 1000.00,
      "spent": 250.50,
      "clicks": 150,
      "impressions": 5000,
      "ctr": 0.03,
      "status": "active",
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 50,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

#### Track Ad Click

```http
POST /ads/{ad_id}/click
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "clicked": true,
  "clicks": 151,
  "message": "Ad click tracked successfully"
}
```

### Payment Endpoints

#### Create Payment Intent

```http
POST /payments/intent
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "amount": 999,
  "currency": "usd",
  "description": "Premium subscription"
}
```

**Response:**
```json
{
  "client_secret": "pi_1234567890_secret_abcdef",
  "payment_intent_id": "pi_1234567890",
  "amount": 999,
  "currency": "usd",
  "status": "requires_payment_method"
}
```

#### Get Payment History

```http
GET /payments/history?skip=0&limit=20
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "payments": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "amount": 999,
      "currency": "usd",
      "description": "Premium subscription",
      "status": "succeeded",
      "payment_method": "card_1234",
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 10,
  "skip": 0,
  "limit": 20,
  "has_more": true
}
```

### Subscription Endpoints

#### Create Subscription

```http
POST /subscriptions
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "plan_id": "premium_monthly",
  "payment_method_id": "pm_1234567890"
}
```

**Response:**
```json
{
  "subscription_id": "sub_1234567890",
  "plan_id": "premium_monthly",
  "status": "active",
  "current_period_start": "2025-01-01T00:00:00Z",
  "current_period_end": "2025-02-01T00:00:00Z",
  "cancel_at_period_end": false,
  "created_at": "2025-01-01T00:00:00Z"
}
```

#### Get Subscription

```http
GET /subscriptions/me
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "subscription_id": "sub_1234567890",
  "plan": {
    "id": "premium_monthly",
    "name": "Premium Monthly",
    "price": 999,
    "currency": "usd",
    "interval": "month",
    "features": [
      "Unlimited uploads",
      "Advanced analytics",
      "Priority support"
    ]
  },
  "status": "active",
  "current_period_start": "2025-01-01T00:00:00Z",
  "current_period_end": "2025-02-01T00:00:00Z",
  "cancel_at_period_end": false,
  "created_at": "2025-01-01T00:00:00Z"
}
```

### Notification Endpoints

#### Get Notifications

```http
GET /notifications?skip=0&limit=20&unread_only=true
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "notifications": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "type": "like",
      "title": "Someone liked your post",
      "message": "nirmalmina liked your post",
      "data": {
        "post_id": "123e4567-e89b-12d3-a456-426614174000",
        "user_id": "123e4567-e89b-12d3-a456-426614174000"
      },
      "is_read": false,
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 25,
  "skip": 0,
  "limit": 20,
  "has_more": true,
  "unread_count": 5
}
```

#### Mark Notification as Read

```http
PUT /notifications/{notification_id}/read
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "read": true,
  "message": "Notification marked as read"
}
```

### Analytics Endpoints

#### Track Event

```http
POST /analytics/track
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "event_type": "video_view",
  "data": {
    "video_id": "123e4567-e89b-12d3-a456-426614174000",
    "duration": 120.5,
    "quality": "720p"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Event tracked successfully"
}
```

#### Get User Analytics

```http
GET /analytics/user/{user_id}?time_period=7d
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "time_period": "7d",
  "metrics": {
    "total_views": 5000,
    "total_likes": 250,
    "total_comments": 50,
    "total_shares": 25,
    "total_followers": 150,
    "total_following": 75,
    "engagement_rate": 0.12,
    "average_watch_time": 15.5,
    "retention_rate": 0.85
  },
  "events_count": 1000,
  "generated_at": "2025-01-15T10:30:00Z"
}
```

### Search Endpoints

#### Search Content

```http
GET /search?q=funny videos&type=mixed&skip=0&limit=20
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "results": [
    {
      "type": "video",
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Funny Cat Video",
      "description": "This cat is hilarious!",
      "thumbnail_url": "https://example.com/thumb.jpg",
      "duration": 60.5,
      "views_count": 10000,
      "likes_count": 500,
      "author": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "username": "nirmalmina",
        "display_name": "Nirmal Mina",
        "avatar_url": "https://example.com/avatar.jpg"
      },
      "created_at": "2025-01-01T00:00:00Z",
      "relevance_score": 0.95
    }
  ],
  "total": 100,
  "skip": 0,
  "limit": 20,
  "has_more": true,
  "query": "funny videos",
  "search_time": 0.05
}
```

### Admin Endpoints

#### Get System Stats

```http
GET /admin/stats
Authorization: Bearer <admin_access_token>
```

**Response:**
```json
{
  "users": {
    "total": 10000,
    "active_today": 5000,
    "new_today": 100,
    "verified": 500
  },
  "content": {
    "videos": {
      "total": 50000,
      "uploaded_today": 500,
      "processed": 45000,
      "processing": 1000
    },
    "posts": {
      "total": 100000,
      "created_today": 1000
    }
  },
  "engagement": {
    "total_views": 1000000,
    "total_likes": 50000,
    "total_comments": 10000,
    "total_shares": 5000
  },
  "revenue": {
    "total": 50000.00,
    "today": 500.00,
    "subscriptions": 30000.00,
    "ads": 20000.00
  },
  "system": {
    "uptime": 86400,
    "cpu_usage": 45.5,
    "memory_usage": 67.8,
    "disk_usage": 23.4
  }
}
```

### Moderation Endpoints

#### Report Content

```http
POST /moderation/reports
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "target_type": "video",
  "target_id": "123e4567-e89b-12d3-a456-426614174000",
  "reason": "inappropriate_content",
  "description": "This video contains inappropriate content"
}
```

**Response:**
```json
{
  "report_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "Report submitted successfully"
}
```

### ML/AI Endpoints

#### Analyze Content

```http
POST /ml/analyze
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

content_type: text
content_data: {"text": "This is a test message"}
```

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Analysis queued successfully"
}
```

#### Get Recommendations

```http
GET /ml/recommendations?content_type=mixed&limit=10
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Recommendations queued successfully"
}
```

#### Get Task Status

```http
GET /ml/tasks/{task_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "result": {
    "sentiment": "positive",
    "toxicity": 0.1,
    "categories": ["entertainment", "funny"],
    "confidence": 0.95
  },
  "created_at": "2025-01-01T00:00:00Z",
  "completed_at": "2025-01-01T00:01:00Z"
}
```

## 🔌 WebSocket Events

### Connection

```javascript
const ws = new WebSocket('wss://api.socialflow.com/ws?token=<access_token>');
```

### Event Types

#### User Events

```json
{
  "type": "user_followed",
  "data": {
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "follower_id": "123e4567-e89b-12d3-a456-426614174000",
    "follower_username": "nirmalmina",
    "follower_display_name": "Nirmal Mina"
  }
}
```

#### Content Events

```json
{
  "type": "video_liked",
  "data": {
    "video_id": "123e4567-e89b-12d3-a456-426614174000",
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "likes_count": 76
  }
}
```

#### Notification Events

```json
{
  "type": "notification",
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "type": "like",
    "title": "Someone liked your post",
    "message": "nirmalmina liked your post",
    "data": {
      "post_id": "123e4567-e89b-12d3-a456-426614174000"
    }
  }
}
```

## 📱 SDK Examples

### JavaScript/TypeScript

```typescript
import { SocialFlowAPI } from '@socialflow/api-client';

const api = new SocialFlowAPI({
  baseURL: 'https://api.socialflow.com/api/v1',
  token: 'your_access_token'
});

// Upload video
const video = await api.videos.upload({
  file: videoFile,
  title: 'My Video',
  description: 'Video description',
  tags: 'funny,comedy'
});

// Get feed
const feed = await api.posts.getFeed({
  skip: 0,
  limit: 20
});

// Like post
await api.likes.create({
  target_type: 'post',
  target_id: 'post_id'
});
```

### Python

```python
from socialflow import SocialFlowAPI

api = SocialFlowAPI(
    base_url='https://api.socialflow.com/api/v1',
    token='your_access_token'
)

# Upload video
video = api.videos.upload(
    file=video_file,
    title='My Video',
    description='Video description',
    tags='funny,comedy'
)

# Get feed
feed = api.posts.get_feed(skip=0, limit=20)

# Like post
api.likes.create(
    target_type='post',
    target_id='post_id'
)
```

### Flutter/Dart

```dart
import 'package:socialflow_api/socialflow_api.dart';

final api = SocialFlowAPI(
  baseURL: 'https://api.socialflow.com/api/v1',
  token: 'your_access_token',
);

// Upload video
final video = await api.videos.upload(
  file: videoFile,
  title: 'My Video',
  description: 'Video description',
  tags: 'funny,comedy',
);

// Get feed
final feed = await api.posts.getFeed(
  skip: 0,
  limit: 20,
);

// Like post
await api.likes.create(
  targetType: 'post',
  targetId: 'post_id',
);
```

## 🔧 Development Tools

### Postman Collection

Import the Postman collection from `docs/api/postman/SocialFlow-API.postman_collection.json`

### OpenAPI Specification

View the complete OpenAPI specification at `openapi.yaml`

### API Explorer

Interactive API documentation available at:
- Development: `http://localhost:8000/docs`
- Production: `https://api.socialflow.com/docs`

## 📞 Support

For API support and questions:
- **Documentation**: [docs.socialflow.com](https://docs.socialflow.com)
- **API Status**: [status.socialflow.com](https://status.socialflow.com)
- **Support Email**: api-support@socialflow.com
- **Discord**: [Join our Discord](https://discord.gg/socialflow)
