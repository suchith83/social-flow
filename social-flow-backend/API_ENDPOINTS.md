# API Endpoints Documentation

This document provides a comprehensive list of all API endpoints for the Social Flow Backend.

## Base URL
```
http://localhost:3000
```

## Authentication
All endpoints (except auth endpoints) require a valid JWT token in the Authorization header:
```
Authorization: Bearer <jwt-token>
```

## Response Format
All responses follow this format:
```json
{
  "success": true,
  "data": {},
  "message": "Success message",
  "timestamp": "2025-01-01T00:00:00.000Z"
}
```

## Error Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message",
    "details": {}
  },
  "timestamp": "2025-01-01T00:00:00.000Z"
}
```

---

## Authentication Endpoints

### Register User
- **POST** `/auth/register`
- **Description**: Register a new user
- **Body**:
  ```json
  {
    "username": "string",
    "email": "string",
    "password": "string",
    "displayName": "string"
  }
  ```
- **Response**: User object with JWT token

### Login User
- **POST** `/auth/login`
- **Description**: Login with email/username and password
- **Body**:
  ```json
  {
    "email": "string",
    "password": "string"
  }
  ```
- **Response**: User object with JWT token

### Refresh Token
- **POST** `/auth/refresh`
- **Description**: Refresh JWT token
- **Body**:
  ```json
  {
    "refreshToken": "string"
  }
  ```
- **Response**: New JWT token

### Get Profile
- **GET** `/auth/profile`
- **Description**: Get current user profile
- **Response**: User object

### Google OAuth Login
- **POST** `/auth/social/google`
- **Description**: Login with Google OAuth
- **Body**:
  ```json
  {
    "code": "string"
  }
  ```
- **Response**: User object with JWT token

### Facebook OAuth Login
- **POST** `/auth/social/facebook`
- **Description**: Login with Facebook OAuth
- **Body**:
  ```json
  {
    "code": "string"
  }
  ```
- **Response**: User object with JWT token

### Twitter OAuth Login
- **POST** `/auth/social/twitter`
- **Description**: Login with Twitter OAuth
- **Body**:
  ```json
  {
    "code": "string"
  }
  ```
- **Response**: User object with JWT token

---

## User Endpoints

### Get User Profile
- **GET** `/users/:id`
- **Description**: Get user profile by ID
- **Response**: User object

### Update User Profile
- **PUT** `/users/:id`
- **Description**: Update user profile
- **Body**:
  ```json
  {
    "displayName": "string",
    "bio": "string",
    "avatarUrl": "string",
    "website": "string",
    "location": "string"
  }
  ```
- **Response**: Updated user object

### Follow User
- **POST** `/users/:id/follow`
- **Description**: Follow a user
- **Response**: Success message

### Unfollow User
- **DELETE** `/users/:id/follow`
- **Description**: Unfollow a user
- **Response**: Success message

### Get User Followers
- **GET** `/users/:id/followers`
- **Description**: Get user's followers
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of user objects

### Get User Following
- **GET** `/users/:id/following`
- **Description**: Get users that the user is following
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of user objects

---

## Video Endpoints

### Upload Video
- **POST** `/videos/upload`
- **Description**: Upload a new video
- **Body**: FormData with video file
- **Response**: Video object

### Get Video
- **GET** `/videos/:id`
- **Description**: Get video details
- **Response**: Video object

### Stream Video
- **GET** `/videos/:id/stream`
- **Description**: Stream video content
- **Response**: Video stream

### Like Video
- **POST** `/videos/:id/like`
- **Description**: Like a video
- **Response**: Success message

### Unlike Video
- **DELETE** `/videos/:id/like`
- **Description**: Unlike a video
- **Response**: Success message

### Comment on Video
- **POST** `/videos/:id/comments`
- **Description**: Add comment to video
- **Body**:
  ```json
  {
    "content": "string"
  }
  ```
- **Response**: Comment object

### Get Video Comments
- **GET** `/videos/:id/comments`
- **Description**: Get video comments
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of comment objects

### Get User Videos
- **GET** `/users/:id/videos`
- **Description**: Get user's videos
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of video objects

---

## Post Endpoints

### Create Post
- **POST** `/posts`
- **Description**: Create a new post
- **Body**:
  ```json
  {
    "content": "string",
    "mediaUrl": "string",
    "hashtags": ["string"]
  }
  ```
- **Response**: Post object

### Get Post
- **GET** `/posts/:id`
- **Description**: Get post details
- **Response**: Post object

### Update Post
- **PUT** `/posts/:id`
- **Description**: Update post
- **Body**:
  ```json
  {
    "content": "string",
    "hashtags": ["string"]
  }
  ```
- **Response**: Updated post object

### Delete Post
- **DELETE** `/posts/:id`
- **Description**: Delete post
- **Response**: Success message

### Like Post
- **POST** `/posts/:id/like`
- **Description**: Like a post
- **Response**: Success message

### Unlike Post
- **DELETE** `/posts/:id/like`
- **Description**: Unlike a post
- **Response**: Success message

### Repost
- **POST** `/posts/:id/repost`
- **Description**: Repost a post
- **Response**: Repost object

### Comment on Post
- **POST** `/posts/:id/comments`
- **Description**: Add comment to post
- **Body**:
  ```json
  {
    "content": "string"
  }
  ```
- **Response**: Comment object

### Get Post Comments
- **GET** `/posts/:id/comments`
- **Description**: Get post comments
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of comment objects

### Get Feed
- **GET** `/posts/feed`
- **Description**: Get user's feed
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of post objects

---

## Search Endpoints

### Search Content
- **POST** `/search`
- **Description**: Search for content
- **Body**:
  ```json
  {
    "query": "string",
    "type": "all|users|videos|posts",
    "page": 1,
    "limit": 20,
    "filters": {},
    "sort": "relevance|date|popularity",
    "order": "asc|desc"
  }
  ```
- **Response**: Search results

### Get Search Suggestions
- **GET** `/search/suggestions`
- **Description**: Get search suggestions
- **Query Parameters**:
  - `q`: string (search query)
  - `type`: string (optional)
- **Response**: Array of suggestions

### Get Trending Hashtags
- **GET** `/search/trending/hashtags`
- **Description**: Get trending hashtags
- **Query Parameters**:
  - `limit`: number (default: 10)
- **Response**: Array of hashtags

### Get Trending Topics
- **GET** `/search/trending/topics`
- **Description**: Get trending topics
- **Query Parameters**:
  - `limit`: number (default: 10)
- **Response**: Array of topics

### Get Recommendations
- **POST** `/search/recommendations`
- **Description**: Get content recommendations
- **Body**:
  ```json
  {
    "userId": "string",
    "type": "videos|posts|users",
    "limit": 20,
    "filters": {}
  }
  ```
- **Response**: Array of recommendations

---

## Analytics Endpoints

### Track Event
- **POST** `/analytics/track`
- **Description**: Track an analytics event
- **Body**:
  ```json
  {
    "type": "PAGE_VIEW|VIDEO_VIEW|POST_VIEW|USER_ACTION|SYSTEM_EVENT|PERFORMANCE|ERROR",
    "category": "USER|CONTENT|TECHNICAL",
    "event": "string",
    "properties": {},
    "context": {}
  }
  ```
- **Response**: Success message

### Get User Analytics
- **GET** `/analytics/user`
- **Description**: Get user analytics
- **Query Parameters**:
  - `startDate`: string (ISO date)
  - `endDate`: string (ISO date)
  - `groupBy`: string (optional)
- **Response**: Analytics data

### Get Video Analytics
- **GET** `/analytics/video/:id`
- **Description**: Get video analytics
- **Query Parameters**:
  - `startDate`: string (ISO date)
  - `endDate`: string (ISO date)
  - `groupBy`: string (optional)
- **Response**: Video analytics data

### Get Analytics Overview
- **GET** `/analytics/overview`
- **Description**: Get analytics overview
- **Response**: Overview data

---

## Admin Endpoints

### Get Admin Stats
- **GET** `/admin/stats`
- **Description**: Get admin statistics
- **Response**: Admin statistics

### Manage User
- **POST** `/admin/users/manage`
- **Description**: Manage user (ban, unban, suspend, etc.)
- **Body**:
  ```json
  {
    "userId": "string",
    "action": "ban|unban|suspend|unsuspend|delete",
    "reason": "string",
    "duration": 7
  }
  ```
- **Response**: Success message

### Moderate Content
- **POST** `/admin/content/moderate`
- **Description**: Moderate content
- **Body**:
  ```json
  {
    "entityType": "video|post|comment|user",
    "entityId": "string",
    "action": "approve|reject|flag|unflag|ban|unban",
    "reason": "string",
    "moderatorId": "string"
  }
  ```
- **Response**: Success message

### Get System Health
- **GET** `/admin/health`
- **Description**: Get system health status
- **Response**: Health status

---

## Notification Endpoints

### Get Notifications
- **GET** `/notifications`
- **Description**: Get user notifications
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of notifications

### Mark Notification as Read
- **PUT** `/notifications/:id/read`
- **Description**: Mark notification as read
- **Response**: Success message

### Mark All Notifications as Read
- **PUT** `/notifications/read-all`
- **Description**: Mark all notifications as read
- **Response**: Success message

---

## Payment Endpoints

### Process Payment
- **POST** `/payments/process`
- **Description**: Process a payment
- **Body**:
  ```json
  {
    "amount": 1000,
    "currency": "USD",
    "paymentMethodId": "string",
    "description": "string"
  }
  ```
- **Response**: Payment object

### Get Payment History
- **GET** `/payments/history`
- **Description**: Get payment history
- **Query Parameters**:
  - `page`: number (default: 1)
  - `limit`: number (default: 20)
- **Response**: Array of payments

### Create Subscription
- **POST** `/payments/subscriptions`
- **Description**: Create a subscription
- **Body**:
  ```json
  {
    "tier": "basic|premium|pro",
    "paymentMethodId": "string"
  }
  ```
- **Response**: Subscription object

### Cancel Subscription
- **DELETE** `/payments/subscriptions/:id`
- **Description**: Cancel subscription
- **Response**: Success message

---

## Ad Endpoints

### Get Ads
- **GET** `/ads`
- **Description**: Get ads for user
- **Query Parameters**:
  - `limit`: number (default: 10)
- **Response**: Array of ads

### Record Ad Impression
- **POST** `/ads/:id/impression`
- **Description**: Record ad impression
- **Response**: Success message

### Record Ad Click
- **POST** `/ads/:id/click`
- **Description**: Record ad click
- **Response**: Success message

---

## Real-time WebSocket Events

### Connection
- **Event**: `connection`
- **Description**: User connects to WebSocket
- **Data**: User ID

### Join Room
- **Event**: `join_room`
- **Data**: `{ roomId: string }`

### Leave Room
- **Event**: `leave_room`
- **Data**: `{ roomId: string }`

### Send Message
- **Event**: `send_message`
- **Data**: `{ roomId: string, message: string }`

### Typing Start
- **Event**: `typing_start`
- **Data**: `{ roomId: string }`

### Typing Stop
- **Event**: `typing_stop`
- **Data**: `{ roomId: string }`

### Video View
- **Event**: `video_view`
- **Data**: `{ videoId: string }`

### Post Like
- **Event**: `post_like`
- **Data**: `{ postId: string }`

### Post Comment
- **Event**: `post_comment`
- **Data**: `{ postId: string, comment: string }`

### User Follow
- **Event**: `user_follow`
- **Data**: `{ followingId: string }`

---

## Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `UNAUTHORIZED` | Authentication required |
| `FORBIDDEN` | Insufficient permissions |
| `NOT_FOUND` | Resource not found |
| `CONFLICT` | Resource already exists |
| `RATE_LIMITED` | Too many requests |
| `INTERNAL_ERROR` | Server error |

---

## Rate Limiting

- **General API**: 100 requests per minute per IP
- **Authentication**: 10 requests per minute per IP
- **Upload**: 5 requests per minute per user
- **Search**: 50 requests per minute per user

---

## Pagination

All list endpoints support pagination with these query parameters:
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)

Response includes:
- `results`: Array of items
- `total`: Total number of items
- `page`: Current page
- `limit`: Items per page
- `totalPages`: Total number of pages
