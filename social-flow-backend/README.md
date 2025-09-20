# Social Flow Backend

A comprehensive social media backend API combining YouTube and Twitter features with advanced AI/ML capabilities. Built with FastAPI, PostgreSQL, Redis, and AWS services.

## 🚀 Features

### Core Features
- **User Management**: Registration, authentication, profiles, social login
- **Video Platform**: Upload, encoding, streaming, live streaming
- **Social Features**: Posts, comments, likes, follows, reposts
- **AI/ML Integration**: Content analysis, recommendations, moderation
- **Analytics**: User engagement, content performance, revenue tracking
- **Monetization**: Advertisements, subscriptions, creator payouts
- **Real-time**: WebSocket connections, live chat, notifications

### Technical Features
- **Scalable Architecture**: Microservices with FastAPI
- **Database**: PostgreSQL with Redis caching
- **Storage**: AWS S3 with CloudFront CDN
- **Video Processing**: AWS MediaConvert with adaptive bitrate streaming
- **AI/ML**: AWS SageMaker integration for content analysis
- **Monitoring**: Comprehensive logging and metrics
- **Security**: JWT authentication, rate limiting, input validation

## 📁 Project Structure

```
social-flow-backend/
├── app/                          # Main FastAPI application
│   ├── api/                      # API endpoints
│   │   └── v1/                   # API version 1
│   │       ├── endpoints/        # Individual endpoint modules
│   │       └── router.py         # Main API router
│   ├── core/                     # Core application components
│   │   ├── config.py             # Configuration management
│   │   ├── database.py           # Database connection
│   │   ├── redis.py              # Redis connection
│   │   ├── logging.py            # Logging configuration
│   │   ├── security.py           # Security utilities
│   │   └── exceptions.py         # Custom exceptions
│   ├── models/                   # SQLAlchemy models
│   ├── schemas/                  # Pydantic schemas
│   ├── services/                 # Business logic services
│   └── workers/                  # Background task workers
├── ai-models/                    # AI/ML model definitions
├── analytics/                    # Analytics processing
├── config/                       # Configuration files
├── docs/                         # Documentation
├── ml-pipelines/                 # ML pipeline definitions
├── monitoring/                   # Monitoring configurations
├── scripts/                      # Utility scripts
├── storage/                      # Storage configurations
├── testing/                      # Test files
├── tools/                        # Development tools
├── workers/                      # Background workers
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Python dependencies
└── openapi.yaml                  # OpenAPI specification
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- Docker (optional)
- AWS CLI (for cloud deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/social-flow-backend.git
   cd social-flow-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up databases**
   ```bash
   # Start PostgreSQL and Redis
   docker-compose up -d postgres redis

   # Run migrations
   alembic upgrade head
   ```

6. **Start the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Admin Panel: http://localhost:8000/admin

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/socialflow
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=socialflow-videos
CLOUDFRONT_DOMAIN=your-cloudfront-domain.cloudfront.net

# ML/AI Configuration
SAGEMAKER_ENDPOINT=your-sagemaker-endpoint
OPENAI_API_KEY=your-openai-api-key

# External Services
STRIPE_SECRET_KEY=your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret
SENDGRID_API_KEY=your-sendgrid-api-key

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

## 📚 API Documentation

### Authentication

All API endpoints (except public ones) require authentication using JWT tokens.

**Register a new user:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "securepassword",
    "display_name": "John Doe"
  }'
```

**Login:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=johndoe&password=securepassword"
```

### Video Upload

**Upload a video:**
```bash
curl -X POST "http://localhost:8000/api/v1/videos/upload" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@video.mp4" \
  -F "title=My Video" \
  -F "description=Video description" \
  -F "tags=funny,comedy"
```

**Get video stream URL:**
```bash
curl -X GET "http://localhost:8000/api/v1/videos/{video_id}/stream?quality=720p" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Live Streaming

**Create live stream:**
```bash
curl -X POST "http://localhost:8000/api/v1/videos/live/create" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "title=Live Stream" \
  -F "description=My live stream"
```

### ML/AI Features

**Analyze content:**
```bash
curl -X POST "http://localhost:8000/api/v1/ml/analyze" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "content_type=text" \
  -F "content_data={\"text\": \"Hello world\"}"
```

**Get recommendations:**
```bash
curl -X GET "http://localhost:8000/api/v1/ml/recommendations?content_type=mixed&limit=10" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Load and stress testing

## 🚀 Deployment

### AWS Deployment

1. **Set up AWS resources using Terraform**
   ```bash
   cd terraform
   terraform init
   terraform plan
   terraform apply
   ```

2. **Deploy application**
   ```bash
   # Build Docker image
   docker build -t socialflow-backend .

   # Push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   docker tag socialflow-backend:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/socialflow-backend:latest
   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/socialflow-backend:latest

   # Deploy to ECS
   aws ecs update-service --cluster socialflow-cluster --service socialflow-service --force-new-deployment
   ```

### Kubernetes Deployment

1. **Apply Kubernetes manifests**
   ```bash
   kubectl apply -f k8s/
   ```

2. **Check deployment status**
   ```bash
   kubectl get pods
   kubectl get services
   ```

## 📊 Monitoring

### Health Checks

- **Application Health**: `GET /health`
- **Database Health**: `GET /health/db`
- **Redis Health**: `GET /health/redis`
- **AWS Services**: `GET /health/aws`

### Metrics

- **Prometheus Metrics**: `GET /metrics`
- **Custom Metrics**: Application-specific metrics
- **Business Metrics**: User engagement, content performance

### Logging

- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Aggregation**: Centralized logging with ELK stack

## 🔒 Security

### Authentication & Authorization
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- OAuth2 social login integration
- Multi-factor authentication (MFA)

### Data Protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection
- Rate limiting and DDoS protection

### Compliance
- GDPR compliance for EU users
- CCPA compliance for California users
- COPPA compliance for users under 13
- SOC 2 Type II compliance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use type hints
- Write clear commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.socialflow.com](https://docs.socialflow.com)
- **API Reference**: [api.socialflow.com/docs](https://api.socialflow.com/docs)
- **Issues**: [GitHub Issues](https://github.com/your-org/social-flow-backend/issues)
- **Discord**: [Join our Discord](https://discord.gg/socialflow)
- **Email**: support@socialflow.com

## 🗺️ Roadmap

### Q1 2024
- [ ] Advanced AI content moderation
- [ ] Real-time collaboration features
- [ ] Enhanced analytics dashboard
- [ ] Mobile app API optimizations

### Q2 2024
- [ ] Blockchain integration for creator payments
- [ ] Advanced recommendation algorithms
- [ ] Multi-language support
- [ ] Advanced streaming features

### Q3 2024
- [ ] AI-powered content creation tools
- [ ] Advanced monetization features
- [ ] Enterprise features
- [ ] Global CDN optimization

## 🙏 Acknowledgments

- FastAPI team for the excellent framework
- PostgreSQL community for the robust database
- AWS for comprehensive cloud services
- Open source contributors and maintainers