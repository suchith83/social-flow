# 📚 Social Flow Backend - Complete API Documentation

> **Comprehensive API Reference for all 107+ endpoints**

---

## 📋 Table of Contents

- [Authentication & User Management (24 endpoints)](#authentication--user-management)
- [Video Platform (32 endpoints)](#video-platform)
- [Social Networking (29 endpoints)](#social-networking)
- [Payments & Monetization (18 endpoints)](#payments--monetization)
- [AI & Machine Learning (15 endpoints)](#ai--machine-learning)
- [Notifications (12 endpoints)](#notifications)
- [Search & Discovery (13 endpoints)](#search--discovery)
- [Moderation & Admin (7 endpoints)](#moderation--admin)
- [Health & Monitoring (3 endpoints)](#health--monitoring)

---

## Authentication & User Management

### 🔐 Authentication Endpoints (9 endpoints)

Base Path: `/api/v1/auth`

#### 1. **Register User**
- **Method:** `POST`
- **Path:** `/api/v1/auth/register`
- **Auth Required:** No
- **Description:** Register a new user account
- **Request Body:**
  ```json
  {
    "email": "user@example.com",
    "username": "johndoe",
    "password": "StrongPassword123!",
    "full_name": "John Doe"
  }
  ```
- **Validation:**
  - Email must be valid and unique
  - Username: 3-50 characters, unique
  - Password: min 8 chars, uppercase, lowercase, digit
- **Response:** `201 Created`
  ```json
  {
    "id": "uuid",
    "email": "user@example.com",
    "username": "johndoe",
    "full_name": "John Doe",
    "created_at": "2025-10-05T10:00:00Z"
  }
  ```

#### 2. **Login (OAuth2 Form)**
- **Method:** `POST`
- **Path:** `/api/v1/auth/login`
- **Auth Required:** No
- **Description:** OAuth2 compatible token login
- **Request Body:** Form data
  ```
  username: user@example.com
  password: StrongPassword123!
  ```
- **Response:** `200 OK`
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 1800,
    "requires_2fa": false
  }
  ```
- **Note:** If 2FA enabled, returns temporary token with `requires_2fa: true`

#### 3. **Login (JSON)**
- **Method:** `POST`
- **Path:** `/api/v1/auth/login/json`
- **Auth Required:** No
- **Description:** JSON-based login alternative
- **Request Body:**
  ```json
  {
    "username_or_email": "user@example.com",
    "password": "StrongPassword123!"
  }
  ```
- **Response:** Same as OAuth2 login

#### 4. **Refresh Token**
- **Method:** `POST`
- **Path:** `/api/v1/auth/refresh`
- **Auth Required:** No (requires refresh token)
- **Description:** Refresh access token using refresh token
- **Request Body:**
  ```json
  {
    "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
  }
  ```
- **Response:** `200 OK`
  ```json
  {
    "access_token": "new_access_token",
    "refresh_token": "new_refresh_token",
    "token_type": "bearer",
    "expires_in": 1800
  }
  ```

#### 5. **Setup 2FA**
- **Method:** `POST`
- **Path:** `/api/v1/auth/2fa/setup`
- **Auth Required:** Yes
- **Description:** Set up two-factor authentication
- **Response:** `200 OK`
  ```json
  {
    "secret": "JBSWY3DPEHPK3PXP",
    "qr_code_url": "otpauth://totp/SocialFlow:user@example.com?...",
    "backup_codes": ["ABC123", "DEF456", ...]
  }
  ```

#### 6. **Verify 2FA Setup**
- **Method:** `POST`
- **Path:** `/api/v1/auth/2fa/verify`
- **Auth Required:** Yes
- **Description:** Verify 2FA token and enable 2FA
- **Request Body:**
  ```json
  {
    "token": "123456"
  }
  ```
- **Response:** `200 OK`
  ```json
  {
    "success": true,
    "message": "Two-factor authentication enabled"
  }
  ```

#### 7. **Login with 2FA**
- **Method:** `POST`
- **Path:** `/api/v1/auth/2fa/login`
- **Auth Required:** No (requires temp token)
- **Description:** Complete login with 2FA token
- **Request Body:**
  ```json
  {
    "token": "temporary_token",
    "two_fa_token": "123456"
  }
  ```
- **Response:** Full access tokens (same as login)

#### 8. **Disable 2FA**
- **Method:** `POST`
- **Path:** `/api/v1/auth/2fa/disable`
- **Auth Required:** Yes
- **Description:** Disable two-factor authentication
- **Request Body:**
  ```json
  {
    "password": "current_password"
  }
  ```
- **Response:** `200 OK`

#### 9. **Get Current User**
- **Method:** `GET`
- **Path:** `/api/v1/auth/me`
- **Auth Required:** Yes
- **Description:** Get current authenticated user info
- **Response:** User profile object

---

### 👤 User Management Endpoints (15 endpoints)

Base Path: `/api/v1/users`

#### 10. **Get Current User Profile**
- **Method:** `GET`
- **Path:** `/api/v1/users/me`
- **Auth Required:** Yes
- **Description:** Get detailed profile of current user
- **Response:** `200 OK` - Complete user profile with stats

#### 11. **Update Current User**
- **Method:** `PUT`
- **Path:** `/api/v1/users/me`
- **Auth Required:** Yes
- **Description:** Update current user's profile
- **Request Body:**
  ```json
  {
    "full_name": "Updated Name",
    "bio": "My bio",
    "website_url": "https://example.com",
    "location": "New York",
    "avatar_url": "https://...",
    "cover_url": "https://..."
  }
  ```
- **Response:** Updated user profile

#### 12. **Change Password**
- **Method:** `PUT`
- **Path:** `/api/v1/users/me/password`
- **Auth Required:** Yes
- **Description:** Change current user's password
- **Request Body:**
  ```json
  {
    "current_password": "old_password",
    "new_password": "NewPassword123!"
  }
  ```
- **Response:** Success message

#### 13. **List Users**
- **Method:** `GET`
- **Path:** `/api/v1/users`
- **Auth Required:** Optional
- **Query Parameters:**
  - `skip`: int (default: 0)
  - `limit`: int (default: 20, max: 100)
  - `role`: UserRole (optional)
  - `status`: UserStatus (optional)
- **Description:** List users with pagination and filtering
- **Response:** Paginated list of user profiles

#### 14. **Search Users**
- **Method:** `GET`
- **Path:** `/api/v1/users/search`
- **Auth Required:** No
- **Query Parameters:**
  - `q`: string (search query, required)
  - `skip`: int
  - `limit`: int
- **Description:** Search users by username, name, or email
- **Response:** Paginated search results

#### 15. **Get User by ID**
- **Method:** `GET`
- **Path:** `/api/v1/users/{user_id}`
- **Auth Required:** Optional
- **Description:** Get public profile of any user
- **Response:** User public profile

#### 16. **Delete User**
- **Method:** `DELETE`
- **Path:** `/api/v1/users/{user_id}`
- **Auth Required:** Yes (self or admin)
- **Description:** Soft delete user account
- **Response:** Success message

#### 17. **Get User Followers**
- **Method:** `GET`
- **Path:** `/api/v1/users/{user_id}/followers`
- **Auth Required:** Optional
- **Query Parameters:** `skip`, `limit`
- **Description:** Get list of users following the specified user
- **Response:** Paginated list of followers

#### 18. **Get User Following**
- **Method:** `GET`
- **Path:** `/api/v1/users/{user_id}/following`
- **Auth Required:** Optional
- **Query Parameters:** `skip`, `limit`
- **Description:** Get list of users that specified user follows
- **Response:** Paginated list of following users

#### 19. **Follow User**
- **Method:** `POST`
- **Path:** `/api/v1/users/{user_id}/follow`
- **Auth Required:** Yes
- **Description:** Follow a user
- **Response:** Follow relationship object

#### 20. **Unfollow User**
- **Method:** `DELETE`
- **Path:** `/api/v1/users/{user_id}/follow`
- **Auth Required:** Yes
- **Description:** Unfollow a user
- **Response:** Success message

#### 21-24. **Admin User Management**
- **Method:** `PUT/POST`
- **Paths:**
  - `/api/v1/users/{user_id}/admin` - Admin update user
  - `/api/v1/users/{user_id}/activate` - Activate user
  - `/api/v1/users/{user_id}/deactivate` - Deactivate user
  - `/api/v1/users/{user_id}/suspend` - Suspend user
- **Auth Required:** Yes (Admin only)
- **Description:** Admin-only user management operations

---

## Video Platform

### 🎥 Video Endpoints (32 endpoints)

Base Path: `/api/v1/videos`

#### 25. **Initiate Video Upload**
- **Method:** `POST`
- **Path:** `/api/v1/videos`
- **Auth Required:** Yes (Creator role)
- **Description:** Create video record and get pre-signed upload URL
- **Request Body:**
  ```json
  {
    "filename": "my-video.mp4",
    "file_size": 104857600,
    "content_type": "video/mp4"
  }
  ```
- **Response:** `201 Created`
  ```json
  {
    "upload_url": "https://s3.amazonaws.com/...",
    "video_id": "uuid",
    "expires_in": 3600
  }
  ```

#### 26. **Complete Video Upload**
- **Method:** `POST`
- **Path:** `/api/v1/videos/{video_id}/complete`
- **Auth Required:** Yes
- **Description:** Update video metadata after upload
- **Request Body:**
  ```json
  {
    "title": "My Awesome Video",
    "description": "Video description",
    "tags": ["tech", "tutorial"],
    "visibility": "public"
  }
  ```
- **Response:** Video object with status "processing"

#### 27. **List Videos**
- **Method:** `GET`
- **Path:** `/api/v1/videos`
- **Auth Required:** Optional
- **Query Parameters:**
  - `skip`: int
  - `limit`: int
  - `sort`: string (published_at, views_count, likes_count, created_at)
- **Description:** List all public videos
- **Response:** Paginated list of videos

#### 28. **Get Trending Videos**
- **Method:** `GET`
- **Path:** `/api/v1/videos/trending`
- **Auth Required:** Optional
- **Query Parameters:**
  - `skip`: int
  - `limit`: int
  - `days`: int (1-30, lookback period)
- **Description:** Get trending videos by view count
- **Response:** Paginated trending videos

#### 29. **Search Videos**
- **Method:** `GET`
- **Path:** `/api/v1/videos/search`
- **Auth Required:** Optional
- **Query Parameters:**
  - `q`: string (search query)
  - `skip`: int
  - `limit`: int
- **Description:** Search videos by title/description
- **Response:** Paginated search results

#### 30. **Get My Videos**
- **Method:** `GET`
- **Path:** `/api/v1/videos/my`
- **Auth Required:** Yes
- **Query Parameters:**
  - `skip`, `limit`
  - `status`: VideoStatus
  - `visibility`: VideoVisibility
- **Description:** Get current user's videos
- **Response:** User's videos with all metadata

#### 31. **Get Video Details**
- **Method:** `GET`
- **Path:** `/api/v1/videos/{video_id}`
- **Auth Required:** Optional
- **Description:** Get detailed video information
- **Visibility Check:** Public/Private/Followers-only
- **Response:** Complete video details

#### 32. **Update Video**
- **Method:** `PUT`
- **Path:** `/api/v1/videos/{video_id}`
- **Auth Required:** Yes (Owner)
- **Description:** Update video metadata
- **Request Body:**
  ```json
  {
    "title": "Updated Title",
    "description": "Updated description",
    "tags": ["updated", "tags"],
    "visibility": "public",
    "allow_comments": true,
    "allow_likes": true,
    "is_monetized": true
  }
  ```
- **Response:** Updated video object

#### 33. **Delete Video**
- **Method:** `DELETE`
- **Path:** `/api/v1/videos/{video_id}`
- **Auth Required:** Yes (Owner or Admin)
- **Description:** Soft delete video (sets status to 'archived')
- **Response:** Success message

#### 34. **Get Streaming URLs**
- **Method:** `GET`
- **Path:** `/api/v1/videos/{video_id}/stream`
- **Auth Required:** Optional
- **Description:** Get HLS/DASH streaming URLs for playback
- **Response:**
  ```json
  {
    "hls_url": "https://cdn.../master.m3u8",
    "dash_url": "https://cdn.../manifest.mpd",
    "thumbnail_url": "https://...",
    "poster_url": "https://...",
    "qualities": {
      "720p": "https://.../720p.m3u8",
      "480p": "https://.../480p.m3u8",
      "360p": "https://.../360p.m3u8"
    }
  }
  ```

#### 35. **Increment Video View**
- **Method:** `POST`
- **Path:** `/api/v1/videos/{video_id}/view`
- **Auth Required:** Optional
- **Description:** Record video view (analytics)
- **Response:** Success message

---

## Social Networking

### 📱 Post Endpoints (29 endpoints)

Base Path: `/api/v1/social`

#### 36. **Create Post**
- **Method:** `POST`
- **Path:** `/api/v1/social/posts`
- **Auth Required:** Yes
- **Description:** Create a new post
- **Request Body:**
  ```json
  {
    "content": "Post content with #hashtags and @mentions",
    "images": ["url1", "url2"],
    "visibility": "public",
    "repost_of_id": null,
    "allow_comments": true,
    "allow_likes": true
  }
  ```
- **Response:** `201 Created` - Post object

#### 37. **List Posts**
- **Method:** `GET`
- **Path:** `/api/v1/social/posts`
- **Auth Required:** Optional
- **Query Parameters:**
  - `skip`, `limit`
  - `user_id`: UUID (optional filter)
- **Description:** List public posts or user-specific posts
- **Response:** Paginated posts

#### 38. **Get Feed**
- **Method:** `GET`
- **Path:** `/api/v1/social/posts/feed`
- **Auth Required:** Yes
- **Query Parameters:** `skip`, `limit`
- **Description:** Get personalized feed from followed users
- **Response:** Paginated feed posts

#### 39. **Get Trending Posts**
- **Method:** `GET`
- **Path:** `/api/v1/social/posts/trending`
- **Auth Required:** Optional
- **Query Parameters:**
  - `skip`, `limit`
  - `days`: int (1-30)
- **Description:** Get trending posts by engagement
- **Response:** Trending posts sorted by (likes + comments*2)

#### 40. **Get Post Details**
- **Method:** `GET`
- **Path:** `/api/v1/social/posts/{post_id}`
- **Auth Required:** Optional
- **Description:** Get detailed post info with interaction flags
- **Response:** Post with is_liked, is_saved, original_post (if repost)

#### 41. **Update Post**
- **Method:** `PUT`
- **Path:** `/api/v1/social/posts/{post_id}`
- **Auth Required:** Yes (Owner)
- **Request Body:** Same as create (partial update)
- **Response:** Updated post

#### 42. **Delete Post**
- **Method:** `DELETE`
- **Path:** `/api/v1/social/posts/{post_id}`
- **Auth Required:** Yes (Owner or Admin)
- **Response:** Success message

#### 43. **Create Comment on Post**
- **Method:** `POST`
- **Path:** `/api/v1/social/posts/{post_id}/comments`
- **Auth Required:** Yes
- **Request Body:**
  ```json
  {
    "content": "Comment text",
    "parent_comment_id": null
  }
  ```
- **Response:** `201 Created` - Comment object

#### 44. **Get Post Comments**
- **Method:** `GET`
- **Path:** `/api/v1/social/posts/{post_id}/comments`
- **Auth Required:** Optional
- **Query Parameters:** `skip`, `limit`
- **Description:** Get top-level comments for post
- **Response:** Paginated comments

#### 45. **Get Comment Details**
- **Method:** `GET`
- **Path:** `/api/v1/social/comments/{comment_id}`
- **Auth Required:** Optional
- **Description:** Get comment with replies (one level)
- **Response:** Comment with replies array

#### 46. **Get Comment Replies**
- **Method:** `GET`
- **Path:** `/api/v1/social/comments/{comment_id}/replies`
- **Auth Required:** Optional
- **Query Parameters:** `skip`, `limit`
- **Response:** Paginated replies

#### 47. **Update Comment**
- **Method:** `PUT`
- **Path:** `/api/v1/social/comments/{comment_id}`
- **Auth Required:** Yes (Owner)
- **Request Body:**
  ```json
  {
    "content": "Updated comment"
  }
  ```
- **Response:** Updated comment (marked as edited)

#### 48. **Delete Comment**
- **Method:** `DELETE`
- **Path:** `/api/v1/social/comments/{comment_id}`
- **Auth Required:** Yes (Owner or Admin)
- **Response:** Success message

#### 49. **Like Post**
- **Method:** `POST`
- **Path:** `/api/v1/social/posts/{post_id}/like`
- **Auth Required:** Yes
- **Response:** Success message

#### 50. **Unlike Post**
- **Method:** `DELETE`
- **Path:** `/api/v1/social/posts/{post_id}/like`
- **Auth Required:** Yes
- **Response:** Success message

#### 51. **Like Comment**
- **Method:** `POST`
- **Path:** `/api/v1/social/comments/{comment_id}/like`
- **Auth Required:** Yes
- **Response:** Success message

#### 52. **Unlike Comment**
- **Method:** `DELETE`
- **Path:** `/api/v1/social/comments/{comment_id}/like`
- **Auth Required:** Yes
- **Response:** Success message

#### 53. **Save Post**
- **Method:** `POST`
- **Path:** `/api/v1/social/posts/{post_id}/save`
- **Auth Required:** Yes
- **Description:** Bookmark post to saved collection
- **Response:** Success message

#### 54. **Unsave Post**
- **Method:** `DELETE`
- **Path:** `/api/v1/social/posts/{post_id}/save`
- **Auth Required:** Yes
- **Response:** Success message

#### 55. **Get Saved Content**
- **Method:** `GET`
- **Path:** `/api/v1/social/saves`
- **Auth Required:** Yes
- **Query Parameters:** `skip`, `limit`
- **Description:** Get user's saved posts and videos
- **Response:** Paginated saved items

#### 56-58. **Admin Moderation**
- **Paths:**
  - `POST /api/v1/social/posts/{post_id}/admin/flag`
  - `POST /api/v1/social/posts/{post_id}/admin/remove`
  - `POST /api/v1/social/comments/{comment_id}/admin/remove`
- **Auth Required:** Yes (Admin)
- **Description:** Admin moderation actions

---

## Payments & Monetization

### 💰 Payment Endpoints (18 endpoints)

Base Path: `/api/v1/payments`

#### 59. **Create Payment Intent**
- **Method:** `POST`
- **Path:** `/api/v1/payments/intent`
- **Auth Required:** Yes
- **Description:** Create Stripe payment intent
- **Request Body:**
  ```json
  {
    "amount": 29.99,
    "currency": "usd",
    "payment_type": "one_time",
    "description": "Video purchase",
    "metadata": {}
  }
  ```
- **Response:** `201 Created`
  ```json
  {
    "payment_id": "uuid",
    "client_secret": "pi_xxx_secret_yyy",
    "stripe_payment_intent_id": "pi_xxx",
    "amount": 29.99,
    "currency": "usd",
    "status": "pending"
  }
  ```

#### 60. **Confirm Payment**
- **Method:** `POST`
- **Path:** `/api/v1/payments/{payment_id}/confirm`
- **Auth Required:** Yes
- **Description:** Confirm payment after Stripe confirmation
- **Response:** Payment object with status "completed"

#### 61. **Refund Payment**
- **Method:** `POST`
- **Path:** `/api/v1/payments/{payment_id}/refund`
- **Auth Required:** Yes
- **Description:** Full or partial refund
- **Request Body:**
  ```json
  {
    "amount": 10.00,
    "reason": "Customer request"
  }
  ```
- **Response:** Updated payment with refund info

#### 62. **List Payments**
- **Method:** `GET`
- **Path:** `/api/v1/payments`
- **Auth Required:** Yes
- **Query Parameters:**
  - `skip`, `limit`
  - `payment_status`: string
- **Description:** Get user's payment history
- **Response:** Paginated payment list

#### 63. **Get Payment Details**
- **Method:** `GET`
- **Path:** `/api/v1/payments/{payment_id}`
- **Auth Required:** Yes
- **Description:** Get detailed payment information
- **Response:** Complete payment object

#### 64. **Create Subscription**
- **Method:** `POST`
- **Path:** `/api/v1/subscriptions`
- **Auth Required:** Yes
- **Description:** Create new subscription
- **Request Body:**
  ```json
  {
    "tier": "premium",
    "trial_days": 7
  }
  ```
- **Response:** `201 Created` - Subscription object

#### 65. **Get Subscription Pricing**
- **Method:** `GET`
- **Path:** `/api/v1/subscriptions/pricing`
- **Auth Required:** No
- **Description:** Get all subscription tiers and pricing
- **Response:** List of pricing tiers with features

#### 66. **Get Current Subscription**
- **Method:** `GET`
- **Path:** `/api/v1/subscriptions/current`
- **Auth Required:** Yes
- **Description:** Get user's active subscription
- **Response:** Subscription details with days remaining

#### 67. **Upgrade Subscription**
- **Method:** `PUT`
- **Path:** `/api/v1/subscriptions/upgrade`
- **Auth Required:** Yes
- **Request Body:**
  ```json
  {
    "subscription_id": "uuid",
    "new_tier": "pro"
  }
  ```
- **Response:** Updated subscription

#### 68. **Cancel Subscription**
- **Method:** `POST`
- **Path:** `/api/v1/subscriptions/cancel`
- **Auth Required:** Yes
- **Request Body:**
  ```json
  {
    "subscription_id": "uuid",
    "immediate": false
  }
  ```
- **Response:** Cancelled subscription

#### 69. **Create Stripe Connect Account**
- **Method:** `POST`
- **Path:** `/api/v1/payouts/connect`
- **Auth Required:** Yes
- **Description:** Setup creator payout account
- **Request Body:**
  ```json
  {
    "account_type": "express",
    "country": "US"
  }
  ```
- **Response:** `201 Created`
  ```json
  {
    "connect_account_id": "acct_xxx",
    "stripe_account_id": "acct_xxx",
    "onboarding_url": "https://connect.stripe.com/setup/...",
    "status": "pending"
  }
  ```

#### 70-76. **Additional Payment Endpoints**
- Get Connect account status
- Request payout
- List payouts
- Get payout details
- Payment analytics
- Subscription analytics
- Creator earnings

---

## AI & Machine Learning

### 🤖 ML Endpoints (15 endpoints)

Base Path: `/api/v1/ml` and `/api/v1/ai`

#### 77. **Get Video Recommendations**
- **Method:** `POST`
- **Path:** `/api/v1/ml/recommendations/videos`
- **Auth Required:** Yes
- **Description:** Get personalized video recommendations
- **Request Body:**
  ```json
  {
    "user_id": "uuid",
    "algorithm": "smart",
    "limit": 20
  }
  ```
- **Algorithms:**
  - `trending` - Engagement-based popularity
  - `collaborative` - Collaborative filtering
  - `content_based` - Content similarity
  - `deep_learning` - Neural network
  - `transformer` - BERT-based (Advanced)
  - `neural_cf` - Neural collaborative filtering (Advanced)
  - `graph` - Graph neural networks (Advanced)
  - `smart` - Multi-armed bandit (Auto-select best)
- **Response:** List of recommended videos with scores

#### 78. **Analyze Video Content**
- **Method:** `POST`
- **Path:** `/api/v1/ml/analyze/video`
- **Auth Required:** Yes
- **Description:** AI-powered video analysis
- **Request Body:**
  ```json
  {
    "video_url": "https://...",
    "analysis_types": ["nsfw", "violence", "objects", "scenes"]
  }
  ```
- **Response:** Comprehensive analysis results

#### 79. **Moderate Content**
- **Method:** `POST`
- **Path:** `/api/v1/ml/moderate/content`
- **Auth Required:** Yes
- **Description:** AI content moderation
- **Request Body:**
  ```json
  {
    "content": "Text to moderate",
    "content_type": "text"
  }
  ```
- **Response:** Moderation results with safety scores

#### 80. **Analyze Sentiment**
- **Method:** `POST`
- **Path:** `/api/v1/ml/sentiment/analyze`
- **Auth Required:** Optional
- **Description:** Sentiment and emotion analysis
- **Request Body:**
  ```json
  {
    "text": "This is amazing!",
    "analyze_emotion": true
  }
  ```
- **Response:**
  ```json
  {
    "sentiment": "positive",
    "confidence": 0.95,
    "emotion": "joy",
    "emotion_confidence": 0.89
  }
  ```

#### 81. **Predict Trends**
- **Method:** `POST`
- **Path:** `/api/v1/ml/trends/predict`
- **Auth Required:** Yes
- **Description:** Predict emerging trends
- **Response:** Trending hashtags and topics with predictions

#### 82-91. **AI Pipeline Endpoints**
- **Base Path:** `/api/v1/ai`
- **Endpoints:**
  - `POST /pipelines/batch-analyze` - Batch video analysis
  - `POST /pipelines/run` - Run specific pipeline
  - `GET /pipelines/status/{pipeline_id}` - Get pipeline status
  - `POST /pipelines/recommendations/warm-cache` - Warm recommendation cache
  - `GET /pipelines/scheduler/jobs` - List scheduled jobs
  - `POST /pipelines/scheduler/start` - Start scheduler
  - `POST /pipelines/scheduler/stop` - Stop scheduler
  - `GET /pipelines/analytics` - Pipeline analytics
  - `GET /pipelines/models` - List available models
  - `POST /pipelines/models/{model_id}/train` - Train model

---

## Notifications

### 🔔 Notification Endpoints (12 endpoints)

Base Path: `/api/v1/notifications`

#### 92. **List Notifications**
- **Method:** `GET`
- **Path:** `/api/v1/notifications`
- **Auth Required:** Yes
- **Query Parameters:**
  - `skip`, `limit`
  - `unread_only`: boolean
  - `notification_type`: string
- **Description:** Get user's notifications
- **Response:** Paginated notifications with unread count

#### 93. **Get Unread Count**
- **Method:** `GET`
- **Path:** `/api/v1/notifications/unread-count`
- **Auth Required:** Yes
- **Description:** Get count of unread notifications
- **Response:**
  ```json
  {
    "unread_count": 5
  }
  ```

#### 94. **Get Notification**
- **Method:** `GET`
- **Path:** `/api/v1/notifications/{notification_id}`
- **Auth Required:** Yes
- **Description:** Get specific notification details
- **Response:** Notification object

#### 95. **Mark as Read**
- **Method:** `POST`
- **Path:** `/api/v1/notifications/{notification_id}/read`
- **Auth Required:** Yes
- **Description:** Mark single notification as read
- **Response:** Updated notification

#### 96. **Mark All as Read**
- **Method:** `POST`
- **Path:** `/api/v1/notifications/mark-all-read`
- **Auth Required:** Yes
- **Description:** Mark all notifications as read
- **Response:** Count of marked notifications

#### 97. **Delete Notification**
- **Method:** `DELETE`
- **Path:** `/api/v1/notifications/{notification_id}`
- **Auth Required:** Yes
- **Description:** Delete notification
- **Response:** Success message

#### 98. **Get Notification Settings**
- **Method:** `GET`
- **Path:** `/api/v1/notifications/settings`
- **Auth Required:** Yes
- **Description:** Get user's notification preferences
- **Response:** Settings object

#### 99. **Update Notification Settings**
- **Method:** `PUT`
- **Path:** `/api/v1/notifications/settings`
- **Auth Required:** Yes
- **Request Body:**
  ```json
  {
    "email_enabled": true,
    "push_enabled": true,
    "in_app_enabled": true,
    "likes_enabled": true,
    "comments_enabled": true,
    "follows_enabled": true
  }
  ```
- **Response:** Updated settings

#### 100. **Register Push Token**
- **Method:** `POST`
- **Path:** `/api/v1/notifications/push-tokens`
- **Auth Required:** Yes
- **Description:** Register FCM/APNS push token
- **Request Body:**
  ```json
  {
    "token": "fcm_token_xxx",
    "device_type": "android",
    "device_name": "My Phone"
  }
  ```
- **Response:** `201 Created` - Push token object

#### 101. **List Push Tokens**
- **Method:** `GET`
- **Path:** `/api/v1/notifications/push-tokens`
- **Auth Required:** Yes
- **Description:** Get user's registered devices
- **Response:** List of push tokens

#### 102. **Delete Push Token**
- **Method:** `DELETE`
- **Path:** `/api/v1/notifications/push-tokens/{token_id}`
- **Auth Required:** Yes
- **Description:** Unregister push token
- **Response:** Success message

---

## Search & Discovery

### 🔍 Search Endpoints (13 endpoints)

Base Path: `/api/v1/search`

#### 103. **Global Search**
- **Method:** `GET`
- **Path:** `/api/v1/search`
- **Auth Required:** Optional
- **Query Parameters:**
  - `q`: string (search query)
  - `type`: string (videos, users, posts, all)
  - `skip`, `limit`
- **Description:** Search across all content types
- **Response:** Combined search results

#### 104. **Search Videos**
- **Method:** `GET`
- **Path:** `/api/v1/search/videos`
- **Auth Required:** Optional
- **Query Parameters:**
  - `q`: string
  - `skip`, `limit`
  - `filters`: object (duration, quality, date)
- **Response:** Video search results

#### 105. **Search Users**
- **Method:** `GET`
- **Path:** `/api/v1/search/users`
- **Auth Required:** Optional
- **Query Parameters:**
  - `q`: string
  - `skip`, `limit`
- **Response:** User search results

#### 106. **Search Hashtags**
- **Method:** `GET`
- **Path:** `/api/v1/search/hashtags`
- **Auth Required:** Optional
- **Query Parameters:**
  - `q`: string
  - `limit`
- **Description:** Search and suggest hashtags
- **Response:** Matching hashtags with post counts

#### 107. **Get Hashtag Analytics**
- **Method:** `GET`
- **Path:** `/api/v1/search/hashtags/{hashtag}/analytics`
- **Auth Required:** Optional
- **Description:** Get hashtag usage statistics
- **Response:** Trend data and post count over time

#### 108-115. **Additional Search Features**
- Search suggestions
- Related content
- Trending searches
- Search history
- Save search
- Recent searches
- Popular hashtags
- Hashtag feed

---

## Moderation & Admin

### 🛡️ Moderation Endpoints (7 endpoints)

Base Path: `/api/v1/moderation` and `/api/v1/admin`

#### 116. **Flag Content**
- **Method:** `POST`
- **Path:** `/api/v1/moderation/flag`
- **Auth Required:** Yes
- **Description:** Flag content for review
- **Request Body:**
  ```json
  {
    "content_type": "post",
    "content_id": "uuid",
    "reason": "spam",
    "details": "Spam content description"
  }
  ```
- **Response:** Flag record

#### 117. **Review Flagged Content**
- **Method:** `GET`
- **Path:** `/api/v1/moderation/flagged`
- **Auth Required:** Yes (Moderator/Admin)
- **Query Parameters:** `skip`, `limit`, `status`
- **Response:** List of flagged content

#### 118-122. **Admin Operations**
- **Paths:**
  - `GET /api/v1/admin/stats` - Platform statistics
  - `GET /api/v1/admin/users` - User management
  - `GET /api/v1/admin/content` - Content management
  - `POST /api/v1/admin/ban-user` - Ban user
  - `POST /api/v1/admin/remove-content` - Remove content

---

## Health & Monitoring

### ✅ Health Endpoints (3 endpoints)

Base Path: `/api/v1`

#### 123. **Health Check**
- **Method:** `GET`
- **Path:** `/health`
- **Auth Required:** No
- **Description:** Basic health check
- **Response:**
  ```json
  {
    "status": "healthy",
    "timestamp": "2025-10-05T10:00:00Z"
  }
  ```

#### 124. **Detailed Health Check**
- **Method:** `GET`
- **Path:** `/health/detailed`
- **Auth Required:** No
- **Description:** Comprehensive health status
- **Response:**
  ```json
  {
    "status": "healthy",
    "database": "connected",
    "redis": "connected",
    "storage": "available",
    "timestamp": "2025-10-05T10:00:00Z"
  }
  ```

#### 125. **Readiness Check**
- **Method:** `GET`
- **Path:** `/health/ready`
- **Auth Required:** No
- **Description:** Kubernetes readiness probe
- **Response:** 200 if ready, 503 if not

---

## 📝 Common Response Patterns

### Success Response
```json
{
  "success": true,
  "message": "Operation completed successfully"
}
```

### Paginated Response
```json
{
  "items": [...],
  "total": 100,
  "skip": 0,
  "limit": 20
}
```

### Error Response
```json
{
  "detail": "Error message",
  "status_code": 400
}
```

---

## 🔑 Authentication

All authenticated endpoints require Bearer token:

```bash
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Token expires in 30 minutes. Use refresh token to get new access token.

---

## 📊 Rate Limiting

- **Default:** 100 requests/minute per user
- **Anonymous:** 20 requests/minute per IP
- **Premium:** 1000 requests/minute

---

## 🌐 Base URL

**Development:** `http://localhost:8000`  
**Production:** `https://api.socialflow.com`

**API Version:** `v1`  
**Full Base:** `http://localhost:8000/api/v1`

---

## 📚 Additional Resources

- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/api/v1/openapi.json
- **Postman Collection:** See `postman_collection.json`

---

## 💡 Quick Examples

### Register and Login
```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","username":"johndoe","password":"Password123!"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login/json \
  -H "Content-Type: application/json" \
  -d '{"username_or_email":"johndoe","password":"Password123!"}'
```

### Upload Video
```bash
# 1. Initiate upload
curl -X POST http://localhost:8000/api/v1/videos \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"filename":"video.mp4","file_size":10485760,"content_type":"video/mp4"}'

# 2. Upload to S3 URL (returned in response)

# 3. Complete upload
curl -X POST http://localhost:8000/api/v1/videos/{video_id}/complete \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"My Video","description":"Great video","visibility":"public"}'
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/api/v1/ml/recommendations/videos \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"smart","limit":20}'
```

---

**Last Updated:** October 5, 2025  
**Version:** 1.0  
**Maintained by:** Social Flow Development Team
