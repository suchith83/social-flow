# 📚 Comprehensive API Documentation

## 🌐 **API Overview**

The Social Flow Backend provides a comprehensive REST API with over 100 endpoints covering all aspects of the social media platform. The API is built with FastAPI and follows RESTful principles with OpenAPI 3.0 specification.

### **🔗 Base URLs**

| **Environment** | **Base URL** | **Description** |
|-----------------|--------------|-----------------|
| **Development** | `http://localhost:8000/api/v1` | Local development |
| **Staging** | `https://api-staging.socialflow.com/api/v1` | Staging environment |
| **Production** | `https://api.socialflow.com/api/v1` | Production environment |

### **📋 API Features**

- **🔐 JWT Authentication**: Secure token-based authentication
- **📊 Rate Limiting**: Configurable rate limiting per endpoint
- **📝 Request/Response Validation**: Automatic validation with Pydantic
- **📖 Auto-generated Documentation**: Interactive API docs with Swagger UI
- **🔄 WebSocket Support**: Real-time features for live streaming and chat
- **📱 Mobile Optimized**: Optimized for mobile app integration
- **🌐 CORS Support**: Cross-origin resource sharing enabled
- **📊 Analytics**: Built-in request/response analytics

---

## 🔐 **Authentication**

### **Authentication Methods**

1. **JWT Bearer Token** (Primary)
2. **OAuth2 Social Login** (Google, Facebook, Twitter)
3. **API Key** (For service-to-service communication)

### **JWT Token Structure**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "uuid",
    "username": "nirmalmina",
    "email": "nirmalmina@socialflow.com",
    "roles": ["user"]
  }
}
```

### **Authentication Endpoints**

#### **POST /auth/register**
Register a new user account.

**Request Body:**
```json
{
  "username": "nirmalmina",
  "email": "nirmalmina@socialflow.com",
  "password": "SecurePassword123!",
  "display_name": "Nirmal Mina",
  "bio": "Software developer and content creator",
  "date_of_birth": "1990-01-01",
  "country": "US",
  "language": "en"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "User registered successfully. Please check your email for verification.",
  "user_id": "uuid",
  "verification_token": "verification_token_here",
  "verification_expires_at": "2024-01-01T12:00:00Z"
}
```

#### **POST /auth/login**
Authenticate user and return JWT tokens.

**Request Body:**
```json
{
  "email": "nirmalmina@socialflow.com",
  "password": "SecurePassword123!"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "uuid",
    "username": "nirmalmina",
    "email": "nirmalmina@socialflow.com",
    "display_name": "Nirmal Mina",
    "avatar_url": "https://cdn.socialflow.com/avatars/uuid.jpg",
    "is_verified": true,
    "roles": ["user"],
    "preferences": {
      "notifications": {
        "email": true,
        "push": true,
        "in_app": true
      },
      "privacy": {
        "profile_visibility": "public",
        "show_online_status": true
      }
    }
  }
}
```

#### **POST /auth/refresh**
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### **POST /auth/logout**
Logout user and invalidate tokens.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### **POST /auth/verify-email**
Verify user email address.

**Request Body:**
```json
{
  "token": "verification_token_here"
}
```

#### **POST /auth/reset-password-request**
Request password reset.

**Request Body:**
```json
{
  "email": "nirmalmina@socialflow.com"
}
```

#### **POST /auth/reset-password**
Reset password with token.

**Request Body:**
```json
{
  "token": "reset_token_here",
  "new_password": "NewSecurePassword123!"
}
```

#### **POST /auth/enable-2fa**
Enable two-factor authentication.

**Response:**
```json
{
  "qr_code": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "secret": "JBSWY3DPEHPK3PXP",
  "backup_codes": [
    "12345678",
    "87654321",
    "11223344"
  ]
}
```

#### **POST /auth/verify-2fa**
Verify 2FA token.

**Request Body:**
```json
{
  "token": "123456"
}
```

#### **GET /auth/profile**
Get current user profile.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "id": "uuid",
  "username": "nirmalmina",
  "email": "nirmalmina@socialflow.com",
  "display_name": "Nirmal Mina",
  "bio": "Software developer and content creator",
  "avatar_url": "https://cdn.socialflow.com/avatars/uuid.jpg",
  "cover_url": "https://cdn.socialflow.com/covers/uuid.jpg",
  "is_verified": true,
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "stats": {
    "followers_count": 1250,
    "following_count": 300,
    "videos_count": 45,
    "posts_count": 120,
    "likes_received": 5000
  },
  "preferences": {
    "notifications": {
      "email": true,
      "push": true,
      "in_app": true
    },
    "privacy": {
      "profile_visibility": "public",
      "show_online_status": true,
      "show_email": false
    }
  }
}
```

---

## 🎥 **Video Management**

### **Video Upload Endpoints**

#### **POST /videos/upload/initiate**
Initiate chunked video upload.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "filename": "my_video.mp4",
  "file_size": 104857600,
  "chunk_size": 1048576,
  "title": "My Amazing Video",
  "description": "This is a description of my video",
  "tags": ["gaming", "tutorial", "funny"],
  "visibility": "public",
  "category": "entertainment",
  "language": "en",
  "thumbnail_timestamp": 30
}
```

**Response:**
```json
{
  "upload_id": "upload_uuid",
  "chunk_count": 100,
  "upload_urls": [
    "https://s3.amazonaws.com/bucket/chunk_1",
    "https://s3.amazonaws.com/bucket/chunk_2"
  ],
  "expires_at": "2024-01-01T13:00:00Z"
}
```

#### **POST /videos/upload/{upload_id}/chunk/{chunk_number}**
Upload video chunk.

**Headers:**
```
Authorization: Bearer <access_token>
Content-Type: application/octet-stream
```

**Request Body:**
```
Binary chunk data
```

#### **POST /videos/upload/{upload_id}/complete**
Complete video upload.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "chunks": [
    {
      "chunk_number": 1,
      "etag": "etag_1"
    },
    {
      "chunk_number": 2,
      "etag": "etag_2"
    }
  ]
}
```

**Response:**
```json
{
  "video_id": "video_uuid",
  "status": "processing",
  "processing_job_id": "job_uuid",
  "estimated_completion": "2024-01-01T12:05:00Z"
}
```

#### **GET /videos/upload/{upload_id}/progress**
Get upload progress.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "upload_id": "upload_uuid",
  "progress": 75,
  "chunks_uploaded": 75,
  "total_chunks": 100,
  "bytes_uploaded": 78643200,
  "total_bytes": 104857600,
  "status": "uploading"
}
```

### **Video Management Endpoints**

#### **GET /videos/{video_id}**
Get video details.

**Response:**
```json
{
  "id": "video_uuid",
  "title": "My Amazing Video",
  "description": "This is a description of my video",
  "filename": "my_video.mp4",
  "duration": 300,
  "file_size": 104857600,
  "resolution": "1920x1080",
  "bitrate": 2000,
  "codec": "h264",
  "thumbnail_url": "https://cdn.socialflow.com/thumbnails/video_uuid.jpg",
  "preview_url": "https://cdn.socialflow.com/previews/video_uuid.mp4",
  "streaming_urls": {
    "hls": "https://cdn.socialflow.com/streams/video_uuid.m3u8",
    "dash": "https://cdn.socialflow.com/streams/video_uuid.mpd"
  },
  "qualities": [
    {
      "quality": "720p",
      "url": "https://cdn.socialflow.com/streams/video_uuid_720p.m3u8",
      "bitrate": 1500
    },
    {
      "quality": "1080p",
      "url": "https://cdn.socialflow.com/streams/video_uuid_1080p.m3u8",
      "bitrate": 3000
    }
  ],
  "owner": {
    "id": "user_uuid",
    "username": "nirmalmina",
    "display_name": "Nirmal Mina",
    "avatar_url": "https://cdn.socialflow.com/avatars/user_uuid.jpg"
  },
  "tags": ["gaming", "tutorial", "funny"],
  "category": "entertainment",
  "language": "en",
  "visibility": "public",
  "is_live": false,
  "status": "published",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "stats": {
    "views": 1250,
    "likes": 45,
    "comments": 12,
    "shares": 8,
    "watch_time": 37500
  }
}
```

#### **PUT /videos/{video_id}**
Update video details.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "title": "Updated Video Title",
  "description": "Updated description",
  "tags": ["gaming", "tutorial", "updated"],
  "visibility": "public",
  "category": "education"
}
```

#### **DELETE /videos/{video_id}**
Delete video.

**Headers:**
```
Authorization: Bearer <access_token>
```

#### **POST /videos/{video_id}/like**
Like a video.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "status": "liked",
  "like_count": 46,
  "is_liked": true
}
```

#### **DELETE /videos/{video_id}/like**
Unlike a video.

**Headers:**
```
Authorization: Bearer <access_token>
```

#### **POST /videos/{video_id}/view**
Record video view.

**Request Body:**
```json
{
  "watch_time": 30,
  "quality": "720p",
  "device_type": "mobile",
  "location": "US"
}
```

#### **GET /videos/feed**
Get personalized video feed.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20, max: 100)
- `category` (str): Filter by category
- `language` (str): Filter by language
- `duration_min` (int): Minimum duration in seconds
- `duration_max` (int): Maximum duration in seconds

**Response:**
```json
{
  "videos": [
    {
      "id": "video_uuid",
      "title": "Video Title",
      "thumbnail_url": "https://cdn.socialflow.com/thumbnails/video_uuid.jpg",
      "duration": 300,
      "views": 1250,
      "likes": 45,
      "created_at": "2024-01-01T12:00:00Z",
      "owner": {
        "id": "user_uuid",
        "username": "nirmalmina",
        "display_name": "Nirmal Mina",
        "avatar_url": "https://cdn.socialflow.com/avatars/user_uuid.jpg"
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1000,
    "pages": 50,
    "has_next": true,
    "has_prev": false
  }
}
```

#### **GET /videos/search**
Search videos.

**Query Parameters:**
- `q` (str): Search query
- `category` (str): Filter by category
- `language` (str): Filter by language
- `duration_min` (int): Minimum duration
- `duration_max` (int): Maximum duration
- `sort` (str): Sort by (relevance, date, views, likes)
- `page` (int): Page number
- `limit` (int): Items per page

#### **POST /videos/{video_id}/transcode**
Request video transcoding.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "quality": "720p",
  "priority": "normal"
}
```

#### **POST /videos/{video_id}/thumbnails**
Generate video thumbnails.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "thumbnails": [
    {
      "time": 10,
      "url": "https://cdn.socialflow.com/thumbnails/video_uuid_10s.jpg"
    },
    {
      "time": 30,
      "url": "https://cdn.socialflow.com/thumbnails/video_uuid_30s.jpg"
    },
    {
      "time": 60,
      "url": "https://cdn.socialflow.com/thumbnails/video_uuid_60s.jpg"
    }
  ]
}
```

---

## 🔴 **Live Streaming**

### **Live Stream Management**

#### **POST /live/start**
Start a new live stream.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "title": "My Live Stream",
  "description": "Live streaming my gameplay",
  "tags": ["gaming", "live", "fun"],
  "visibility": "public",
  "chat_enabled": true,
  "recording_enabled": true,
  "thumbnail_url": "https://cdn.socialflow.com/thumbnails/stream_uuid.jpg"
}
```

**Response:**
```json
{
  "stream_id": "stream_uuid",
  "title": "My Live Stream",
  "stream_key": "live_abc123def456",
  "ingest_url": "rtmp://ingest.socialflow.com/live/live_abc123def456",
  "playback_url": "https://stream.socialflow.com/live/stream_uuid.m3u8",
  "chat_websocket_url": "wss://chat.socialflow.com/ws/stream_uuid",
  "status": "live",
  "viewer_count": 0,
  "started_at": "2024-01-01T12:00:00Z"
}
```

#### **POST /live/{stream_id}/end**
End a live stream.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "stream_id": "stream_uuid",
  "status": "ended",
  "ended_at": "2024-01-01T13:00:00Z",
  "duration": 3600,
  "total_viewers": 150,
  "peak_viewers": 75,
  "recorded_video_id": "video_uuid"
}
```

#### **GET /live/{stream_id}**
Get live stream details.

**Response:**
```json
{
  "id": "stream_uuid",
  "title": "My Live Stream",
  "description": "Live streaming my gameplay",
  "status": "live",
  "viewer_count": 25,
  "started_at": "2024-01-01T12:00:00Z",
  "owner": {
    "id": "user_uuid",
    "username": "nirmalmina",
    "display_name": "Nirmal Mina",
    "avatar_url": "https://cdn.socialflow.com/avatars/user_uuid.jpg"
  },
  "playback_url": "https://stream.socialflow.com/live/stream_uuid.m3u8",
  "chat_enabled": true,
  "recording_enabled": true
}
```

#### **POST /live/{stream_id}/join**
Join a live stream as viewer.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "status": "joined",
  "viewer_id": "viewer_uuid",
  "chat_websocket_url": "wss://chat.socialflow.com/ws/stream_uuid"
}
```

#### **POST /live/{stream_id}/leave**
Leave a live stream.

**Headers:**
```
Authorization: Bearer <access_token>
```

#### **GET /live/{stream_id}/viewers**
Get live stream viewers.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `limit` (int): Number of viewers to return
- `offset` (int): Offset for pagination

**Response:**
```json
{
  "viewers": [
    {
      "id": "viewer_uuid",
      "user": {
        "id": "user_uuid",
        "username": "viewer1",
        "display_name": "Viewer One",
        "avatar_url": "https://cdn.socialflow.com/avatars/user_uuid.jpg"
      },
      "joined_at": "2024-01-01T12:05:00Z",
      "watch_duration": 300
    }
  ],
  "total_viewers": 25,
  "anonymous_viewers": 5
}
```

#### **POST /live/{stream_id}/chat**
Send chat message.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "message": "Hello everyone!",
  "type": "text"
}
```

**Response:**
```json
{
  "message_id": "message_uuid",
  "message": "Hello everyone!",
  "type": "text",
  "timestamp": "2024-01-01T12:05:00Z",
  "user": {
    "id": "user_uuid",
    "username": "nirmalmina",
    "display_name": "Nirmal Mina",
    "avatar_url": "https://cdn.socialflow.com/avatars/user_uuid.jpg"
  }
}
```

#### **GET /live/{stream_id}/chat**
Get chat history.

**Query Parameters:**
- `limit` (int): Number of messages to return
- `offset` (int): Offset for pagination

#### **GET /live/active**
Get active live streams.

**Query Parameters:**
- `category` (str): Filter by category
- `language` (str): Filter by language
- `limit` (int): Number of streams to return
- `offset` (int): Offset for pagination

---

## 🤖 **AI/ML Features**

### **Recommendation System**

#### **GET /ml/recommendations/{user_id}**
Get personalized recommendations.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `type` (str): Recommendation type (videos, users, content)
- `limit` (int): Number of recommendations
- `category` (str): Filter by category

**Response:**
```json
{
  "recommendations": [
    {
      "id": "video_uuid",
      "type": "video",
      "title": "Recommended Video",
      "thumbnail_url": "https://cdn.socialflow.com/thumbnails/video_uuid.jpg",
      "score": 0.95,
      "reason": "Based on your viewing history",
      "metadata": {
        "category": "gaming",
        "duration": 300,
        "views": 5000
      }
    }
  ],
  "algorithm": "hybrid",
  "generated_at": "2024-01-01T12:00:00Z"
}
```

#### **POST /ml/feedback**
Submit user feedback for recommendations.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "content_id": "video_uuid",
  "content_type": "video",
  "feedback_type": "like",
  "rating": 5,
  "context": {
    "session_id": "session_uuid",
    "device_type": "mobile",
    "location": "US"
  }
}
```

### **Content Moderation**

#### **POST /ml/moderate**
Moderate content for safety.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "content_type": "video",
  "content_id": "video_uuid",
  "content_data": {
    "text": "Video description text",
    "title": "Video title",
    "tags": ["gaming", "funny"]
  }
}
```

**Response:**
```json
{
  "content_id": "video_uuid",
  "is_safe": true,
  "confidence": 0.92,
  "flags": [],
  "suggestions": [
    "Consider adding more descriptive tags"
  ],
  "moderation_score": 0.15,
  "categories": {
    "violence": 0.05,
    "adult_content": 0.02,
    "hate_speech": 0.01,
    "spam": 0.07
  }
}
```

### **Trending Analysis**

#### **GET /ml/trending**
Get trending content analysis.

**Query Parameters:**
- `time_range` (str): Time range (1h, 24h, 7d, 30d)
- `category` (str): Filter by category
- `limit` (int): Number of items

**Response:**
```json
{
  "trending_topics": [
    {
      "topic": "gaming",
      "score": 0.95,
      "growth": 0.15,
      "volume": 10000
    }
  ],
  "trending_hashtags": [
    {
      "hashtag": "#gaming",
      "mentions": 5000,
      "growth": 0.12
    }
  ],
  "trending_creators": [
    {
      "id": "user_uuid",
      "username": "gamer123",
      "growth": 0.08,
      "follower_increase": 500
    }
  ],
  "analysis_period": "24h",
  "generated_at": "2024-01-01T12:00:00Z"
}
```

---

## 💳 **Payment & Monetization**

### **Payment Processing**

#### **POST /payments/process**
Process a payment.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "amount": 1000,
  "currency": "USD",
  "payment_method": "stripe",
  "payment_method_id": "pm_1234567890",
  "description": "Premium subscription",
  "metadata": {
    "subscription_plan": "premium",
    "billing_cycle": "monthly"
  }
}
```

**Response:**
```json
{
  "payment_id": "payment_uuid",
  "status": "succeeded",
  "amount": 1000,
  "currency": "USD",
  "payment_method": "stripe",
  "transaction_id": "txn_1234567890",
  "created_at": "2024-01-01T12:00:00Z",
  "receipt_url": "https://pay.stripe.com/receipts/..."
}
```

#### **GET /payments/history**
Get payment history.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `page` (int): Page number
- `limit` (int): Items per page
- `status` (str): Filter by status
- `date_from` (str): Start date
- `date_to` (str): End date

### **Subscription Management**

#### **GET /subscriptions/plans**
Get available subscription plans.

**Response:**
```json
{
  "plans": [
    {
      "id": "plan_uuid",
      "name": "Basic",
      "description": "Basic features",
      "price": 999,
      "currency": "USD",
      "billing_cycle": "monthly",
      "features": [
        "HD video uploads",
        "Basic analytics",
        "Standard support"
      ],
      "limits": {
        "storage_gb": 10,
        "video_uploads_per_month": 50
      }
    },
    {
      "id": "plan_uuid",
      "name": "Premium",
      "description": "Premium features",
      "price": 1999,
      "currency": "USD",
      "billing_cycle": "monthly",
      "features": [
        "4K video uploads",
        "Advanced analytics",
        "Priority support",
        "Live streaming"
      ],
      "limits": {
        "storage_gb": 100,
        "video_uploads_per_month": 500
      }
    }
  ]
}
```

#### **POST /subscriptions/subscribe**
Subscribe to a plan.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "plan_id": "plan_uuid",
  "payment_method_id": "pm_1234567890",
  "billing_cycle": "monthly"
}
```

#### **PUT /subscriptions/{subscription_id}**
Update subscription.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "plan_id": "new_plan_uuid",
  "billing_cycle": "yearly"
}
```

#### **DELETE /subscriptions/{subscription_id}**
Cancel subscription.

**Headers:**
```
Authorization: Bearer <access_token>
```

### **Creator Earnings**

#### **GET /payments/earnings**
Get creator earnings.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `time_range` (str): Time range (7d, 30d, 90d, 1y)
- `page` (int): Page number
- `limit` (int): Items per page

**Response:**
```json
{
  "earnings": {
    "total": 50000,
    "currency": "USD",
    "time_range": "30d",
    "breakdown": {
      "ad_revenue": 30000,
      "subscriptions": 15000,
      "donations": 5000
    }
  },
  "transactions": [
    {
      "id": "transaction_uuid",
      "type": "ad_revenue",
      "amount": 1000,
      "currency": "USD",
      "description": "Video ad revenue",
      "date": "2024-01-01T12:00:00Z"
    }
  ],
  "payouts": [
    {
      "id": "payout_uuid",
      "amount": 50000,
      "currency": "USD",
      "status": "pending",
      "scheduled_date": "2024-01-15T00:00:00Z"
    }
  ]
}
```

---

## 🔔 **Notifications**

### **Notification Management**

#### **GET /notifications/**
Get user notifications.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `page` (int): Page number
- `limit` (int): Items per page
- `type` (str): Filter by type
- `unread_only` (bool): Show only unread

**Response:**
```json
{
  "notifications": [
    {
      "id": "notification_uuid",
      "type": "like",
      "title": "New Like",
      "message": "Nirmal Mina liked your video",
      "data": {
        "video_id": "video_uuid",
        "user_id": "user_uuid"
      },
      "is_read": false,
      "created_at": "2024-01-01T12:00:00Z"
    }
  ],
  "unread_count": 5,
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "pages": 5
  }
}
```

#### **POST /notifications/{notification_id}/read**
Mark notification as read.

**Headers:**
```
Authorization: Bearer <access_token>
```

#### **POST /notifications/read-all**
Mark all notifications as read.

**Headers:**
```
Authorization: Bearer <access_token>
```

#### **GET /notifications/preferences**
Get notification preferences.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "email_enabled": true,
  "push_enabled": true,
  "in_app_enabled": true,
  "types": {
    "likes": true,
    "comments": true,
    "follows": true,
    "mentions": true,
    "ads": false,
    "system": true
  }
}
```

#### **PUT /notifications/preferences**
Update notification preferences.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "email_enabled": true,
  "push_enabled": false,
  "in_app_enabled": true,
  "types": {
    "likes": true,
    "comments": false,
    "follows": true,
    "mentions": true,
    "ads": false,
    "system": true
  }
}
```

---

## 📊 **Analytics**

### **Analytics Endpoints**

#### **POST /analytics/track**
Track analytics event.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "event_type": "video_view",
  "entity_type": "video",
  "entity_id": "video_uuid",
  "properties": {
    "duration": 30,
    "quality": "720p",
    "device_type": "mobile"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### **GET /analytics/dashboard**
Get analytics dashboard data.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `time_range` (str): Time range (7d, 30d, 90d, 1y)
- `metrics` (str): Comma-separated metrics

**Response:**
```json
{
  "overview": {
    "total_views": 100000,
    "total_likes": 5000,
    "total_comments": 1000,
    "total_shares": 500,
    "total_followers": 1000,
    "engagement_rate": 0.065
  },
  "content_performance": {
    "top_videos": [
      {
        "id": "video_uuid",
        "title": "Top Video",
        "views": 10000,
        "likes": 500,
        "engagement_rate": 0.05
      }
    ],
    "top_posts": [
      {
        "id": "post_uuid",
        "content": "Top Post",
        "likes": 200,
        "comments": 50,
        "shares": 25
      }
    ]
  },
  "audience_insights": {
    "demographics": {
      "age_groups": {
        "18-24": 0.3,
        "25-34": 0.4,
        "35-44": 0.2,
        "45+": 0.1
      },
      "locations": {
        "US": 0.4,
        "UK": 0.2,
        "CA": 0.15,
        "AU": 0.1,
        "Other": 0.15
      }
    },
    "engagement_patterns": {
      "peak_hours": [18, 19, 20, 21],
      "peak_days": ["Friday", "Saturday", "Sunday"]
    }
  },
  "revenue_analytics": {
    "total_revenue": 5000,
    "currency": "USD",
    "revenue_sources": {
      "ads": 3000,
      "subscriptions": 1500,
      "donations": 500
    },
    "growth_rate": 0.15
  }
}
```

#### **GET /analytics/reports**
Generate analytics reports.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `report_type` (str): Report type (content, audience, revenue)
- `format` (str): Export format (json, csv, pdf)
- `date_from` (str): Start date
- `date_to` (str): End date

---

## 🔍 **Search**

### **Search Endpoints**

#### **GET /search/content**
Search content (videos, posts).

**Query Parameters:**
- `q` (str): Search query
- `type` (str): Content type (videos, posts, all)
- `category` (str): Filter by category
- `language` (str): Filter by language
- `sort` (str): Sort by (relevance, date, views, likes)
- `page` (int): Page number
- `limit` (int): Items per page

**Response:**
```json
{
  "results": [
    {
      "id": "video_uuid",
      "type": "video",
      "title": "Search Result Video",
      "description": "Video description",
      "thumbnail_url": "https://cdn.socialflow.com/thumbnails/video_uuid.jpg",
      "duration": 300,
      "views": 1000,
      "likes": 50,
      "created_at": "2024-01-01T12:00:00Z",
      "owner": {
        "id": "user_uuid",
        "username": "nirmalmina",
        "display_name": "Nirmal Mina"
      },
      "relevance_score": 0.95
    }
  ],
  "suggestions": [
    "gaming tutorial",
    "funny videos",
    "tech reviews"
  ],
  "filters": {
    "categories": ["gaming", "tech", "entertainment"],
    "languages": ["en", "es", "fr"],
    "date_ranges": ["today", "week", "month", "year"]
  },
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1000,
    "pages": 50
  }
}
```

#### **GET /search/users**
Search users.

**Query Parameters:**
- `q` (str): Search query
- `verified_only` (bool): Show only verified users
- `page` (int): Page number
- `limit` (int): Items per page

#### **GET /search/hashtags**
Search hashtags.

**Query Parameters:**
- `q` (str): Search query
- `trending` (bool): Show trending hashtags
- `limit` (int): Items per page

---

## 🛡️ **Moderation**

### **Content Moderation**

#### **POST /moderation/report**
Report content.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "content_type": "video",
  "content_id": "video_uuid",
  "reason": "inappropriate_content",
  "description": "Contains violent content",
  "evidence_urls": ["https://example.com/evidence.jpg"]
}
```

#### **GET /moderation/reports**
Get moderation reports (admin only).

**Headers:**
```
Authorization: Bearer <access_token>
```

#### **POST /moderation/reports/{report_id}/review**
Review moderation report (admin only).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "action": "remove_content",
  "reason": "Violates community guidelines",
  "notify_user": true
}
```

---

## 👨‍💼 **Admin**

### **Admin Endpoints**

#### **GET /admin/users**
Get all users (admin only).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `page` (int): Page number
- `limit` (int): Items per page
- `status` (str): Filter by status
- `verified` (bool): Filter by verification status

#### **POST /admin/users/{user_id}/ban**
Ban user (admin only).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Request Body:**
```json
{
  "reason": "Violation of terms of service",
  "duration": 7,
  "notify_user": true
}
```

#### **GET /admin/analytics**
Get platform analytics (admin only).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Query Parameters:**
- `time_range` (str): Time range
- `metrics` (str): Comma-separated metrics

---

## 📱 **WebSocket Endpoints**

### **Real-time Features**

#### **WebSocket Connection**
```
wss://api.socialflow.com/ws/{user_id}?token={access_token}
```

#### **Live Chat**
```
wss://chat.socialflow.com/ws/{stream_id}?token={access_token}
```

#### **Real-time Notifications**
```
wss://notifications.socialflow.com/ws/{user_id}?token={access_token}
```

---

## 📊 **Rate Limiting**

### **Rate Limits**

| **Endpoint Category** | **Rate Limit** | **Burst** |
|----------------------|----------------|-----------|
| **Authentication** | 10 req/min | 20 |
| **Video Upload** | 5 req/min | 10 |
| **Search** | 60 req/min | 120 |
| **General API** | 100 req/min | 200 |
| **Admin** | 200 req/min | 400 |

### **Rate Limit Headers**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## 🔒 **Error Handling**

### **Error Response Format**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    },
    "request_id": "req_uuid",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### **HTTP Status Codes**

| **Code** | **Description** |
|----------|-----------------|
| **200** | Success |
| **201** | Created |
| **400** | Bad Request |
| **401** | Unauthorized |
| **403** | Forbidden |
| **404** | Not Found |
| **422** | Validation Error |
| **429** | Rate Limited |
| **500** | Internal Server Error |

---

## 📋 **SDK Examples**

### **Python SDK**

```python
from social_flow import SocialFlowClient

client = SocialFlowClient(
    api_key="your_api_key",
    base_url="https://api.socialflow.com"
)

# Upload video
video = client.videos.upload(
    file_path="video.mp4",
    title="My Video",
    description="Video description"
)

# Get recommendations
recommendations = client.ml.get_recommendations(
    user_id="user_uuid",
    type="videos",
    limit=10
)
```

### **JavaScript SDK**

```javascript
import { SocialFlowClient } from '@social-flow/sdk';

const client = new SocialFlowClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.socialflow.com'
});

// Upload video
const video = await client.videos.upload({
  file: videoFile,
  title: 'My Video',
  description: 'Video description'
});

// Get recommendations
const recommendations = await client.ml.getRecommendations({
  userId: 'user_uuid',
  type: 'videos',
  limit: 10
});
```

---

## 🧪 **Testing**

### **Test Environment**

- **Base URL**: `https://api-test.socialflow.com/api/v1`
- **Test Data**: Sandbox environment with test data
- **Rate Limits**: Higher limits for testing

### **Test Credentials**

```json
{
  "test_user": {
    "email": "test@socialflow.com",
    "password": "TestPassword123!",
    "api_key": "test_api_key_123"
  }
}
```

---

## 📞 **Support**

### **API Support**

- **Documentation**: [https://docs.socialflow.com](https://docs.socialflow.com)
- **Status Page**: [https://status.socialflow.com](https://status.socialflow.com)
- **Support Email**: api-support@socialflow.com
- **Discord**: [https://discord.gg/socialflow](https://discord.gg/socialflow)

### **Rate Limit Support**

- **Rate Limit Issues**: Contact support for rate limit increases
- **Bulk Operations**: Use batch endpoints for bulk operations
- **High Volume**: Consider enterprise plan for high-volume usage
