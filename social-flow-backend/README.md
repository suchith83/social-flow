# Social Flow Backend

A production-grade backend for a social media platform combining features of YouTube, Twitter, and other modern platforms. Built with NestJS, TypeScript, and AWS services.

## Features

- **User Authentication & Authorization**: JWT, OAuth2, social login (Google, Facebook, Twitter)
- **Video Processing**: Upload, encoding, storage, and streaming (like YouTube)
- **Social Features**: Posts, comments, likes, retweets/reposts (like Twitter)
- **Monetization**: Advertisements system with targeting & revenue sharing
- **Payments & Subscriptions**: Stripe/PayPal + in-app purchases support
- **Notifications**: Push, email, in-app notifications
- **Real-time Updates**: WebSockets for live interactions
- **Search & Recommendations**: Elasticsearch + ML models
- **Analytics**: View counts, impressions, click-through rates, retention, watch time
- **Moderation Tools**: Reporting, flagging, AI content moderation
- **Admin Dashboard**: Stats, user bans, ad approvals, system health

## Tech Stack

- **Framework**: NestJS with TypeScript
- **Database**: PostgreSQL (primary), DynamoDB (fast lookups)
- **Cache**: Redis
- **Search**: AWS Elasticsearch (OpenSearch)
- **Storage**: AWS S3 + CloudFront CDN
- **Video Processing**: AWS MediaConvert
- **Authentication**: AWS Cognito (optional) + JWT
- **Queue**: BullMQ (Redis-based)
- **Real-time**: Socket.IO + Redis Pub/Sub
- **Monitoring**: AWS CloudWatch + X-Ray
- **Notifications**: AWS SNS + SES

## Prerequisites

- Node.js 18+ and npm
- PostgreSQL 13+
- Redis 6+
- AWS Account with appropriate services enabled

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
cd social-flow-backend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment Setup**
   ```bash
   cp .env.example .env
   ```
   
   Fill in the environment variables in `.env`:
   ```env
   # App Configuration
   NODE_ENV=development
   APP_PORT=3000
   JWT_SECRET=your-jwt-secret
   JWT_EXPIRES_IN=7d
   
   # Database
   DATABASE_HOST=localhost
   DATABASE_PORT=5432
   DATABASE_USERNAME=postgres
   DATABASE_PASSWORD=password
   DATABASE_NAME=social_flow
   DATABASE_SSL=false
   
   # Redis
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=
   
   # AWS Configuration
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   S3_BUCKET_NAME=social-flow-videos
   S3_BUCKET_REGION=us-east-1
   MEDIACONVERT_ENDPOINT=https://your-mediaconvert-endpoint
   
   # Social Login
   GOOGLE_CLIENT_ID=your-google-client-id
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   FACEBOOK_CLIENT_ID=your-facebook-client-id
   FACEBOOK_CLIENT_SECRET=your-facebook-client-secret
   TWITTER_CLIENT_ID=your-twitter-client-id
   TWITTER_CLIENT_SECRET=your-twitter-client-secret
   
   # Stripe
   STRIPE_SECRET_KEY=your-stripe-secret-key
   STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret
   
   # Email
   AWS_SES_REGION=us-east-1
   FROM_EMAIL=noreply@yourdomain.com
   ```

4. **Database Setup**
   ```bash
   # Create PostgreSQL database
   createdb social_flow
   
   # Run migrations (if available)
   npm run migration:run
   ```

5. **Start the application**
   ```bash
   # Development
   npm run start:dev
   
   # Production
   npm run build
   npm run start:prod
   ```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:3000/api`
- Health Check: `http://localhost:3000/health`

## Project Structure

```
src/
├── auth/                    # Authentication & authorization
├── users/                   # User management
├── videos/                  # Video processing & streaming
├── posts/                   # Social posts & interactions
├── ads/                     # Advertisement system
├── payments/                # Payment processing
├── notifications/           # Notification system
├── analytics/               # Analytics & metrics
├── admin/                   # Admin dashboard
├── search/                  # Search & recommendations
├── realtime/                # WebSocket real-time features
├── moderation/              # Content moderation
└── shared/                  # Shared utilities & services
    ├── config/              # Configuration files
    ├── database/            # Database entities & repositories
    ├── redis/               # Redis service
    ├── aws/                 # AWS services
    ├── logger/              # Logging service
    ├── utils/               # Utility functions
    └── middleware/          # Custom middleware
```

## Key API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Refresh JWT token
- `GET /auth/profile` - Get user profile
- `POST /auth/social/google` - Google OAuth login
- `POST /auth/social/facebook` - Facebook OAuth login
- `POST /auth/social/twitter` - Twitter OAuth login

### Users
- `GET /users/:id` - Get user profile
- `PUT /users/:id` - Update user profile
- `POST /users/:id/follow` - Follow user
- `DELETE /users/:id/follow` - Unfollow user
- `GET /users/:id/followers` - Get user followers
- `GET /users/:id/following` - Get user following

### Videos
- `POST /videos/upload` - Upload video
- `GET /videos/:id` - Get video details
- `GET /videos/:id/stream` - Stream video
- `POST /videos/:id/like` - Like video
- `DELETE /videos/:id/like` - Unlike video
- `POST /videos/:id/comment` - Comment on video
- `GET /videos/:id/comments` - Get video comments

### Posts
- `POST /posts` - Create post
- `GET /posts/:id` - Get post details
- `PUT /posts/:id` - Update post
- `DELETE /posts/:id` - Delete post
- `POST /posts/:id/like` - Like post
- `POST /posts/:id/repost` - Repost
- `POST /posts/:id/comment` - Comment on post

### Search
- `POST /search` - Search content
- `GET /search/suggestions` - Get search suggestions
- `GET /search/trending/hashtags` - Get trending hashtags
- `POST /search/recommendations` - Get recommendations

### Analytics
- `POST /analytics/track` - Track analytics event
- `GET /analytics/user` - Get user analytics
- `GET /analytics/video/:id` - Get video analytics
- `GET /analytics/overview` - Get analytics overview

### Admin
- `GET /admin/stats` - Get admin statistics
- `POST /admin/users/manage` - Manage users
- `POST /admin/content/moderate` - Moderate content
- `GET /admin/health` - Get system health

## Development

### Running Tests
```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Test coverage
npm run test:cov
```

### Code Quality
```bash
# Linting
npm run lint

# Formatting
npm run format

# Type checking
npm run type-check
```

### Database Migrations
```bash
# Generate migration
npm run migration:generate -- -n MigrationName

# Run migrations
npm run migration:run

# Revert migration
npm run migration:revert
```

## Deployment

### AWS Deployment

1. **Infrastructure Setup**
   - Use Terraform or AWS CDK to provision infrastructure
   - Set up RDS PostgreSQL instance
   - Configure ElastiCache Redis cluster
   - Set up S3 buckets for video storage
   - Configure CloudFront distribution

2. **Application Deployment**
   - Build Docker image
   - Deploy to ECS or EKS
   - Configure load balancer
   - Set up auto-scaling

3. **Environment Variables**
   - Set production environment variables
   - Configure AWS IAM roles
   - Set up secrets management

### Docker Deployment

```bash
# Build image
docker build -t social-flow-backend .

# Run container
docker run -p 3000:3000 --env-file .env social-flow-backend
```

## Monitoring & Logging

- **Application Logs**: Winston logger with structured logging
- **Metrics**: Prometheus metrics collection
- **Tracing**: AWS X-Ray distributed tracing
- **Health Checks**: Built-in health check endpoints
- **Error Tracking**: Centralized error logging and alerting

## Security

- **Authentication**: JWT tokens with refresh mechanism
- **Authorization**: Role-based access control (RBAC)
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: API rate limiting with Redis
- **CORS**: Configurable CORS policies
- **HTTPS**: TLS encryption for all communications
- **Secrets Management**: AWS Secrets Manager integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## Roadmap

- [ ] GraphQL API support
- [ ] Advanced ML recommendations
- [ ] Live streaming capabilities
- [ ] Mobile app push notifications
- [ ] Advanced analytics dashboard
- [ ] Content moderation AI
- [ ] Multi-language support
- [ ] API versioning
- [ ] Advanced caching strategies
- [ ] Microservices architecture