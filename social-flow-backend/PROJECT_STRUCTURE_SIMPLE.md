# 📁 **Social Flow Backend - Project Structure**

## 🏗️ **Main Architecture**

```
social-flow-backend/
├── 📱 app/                          # FastAPI Application
│   ├── 🔧 core/                     # Core configuration
│   ├── 📊 models/                   # Database models
│   ├── 🔌 api/                      # API endpoints
│   ├── 🛠️ services/                # Business logic
│   └── 📋 schemas/                  # Data schemas
├── 🧪 tests/                        # Test suite
├── 📚 docs/                         # Documentation
├── 🚀 deployment/                   # Deployment configs
└── 📄 Configuration files
```

## 🔧 **Core Components**

### **App Structure (`app/`)**
- `main.py` - FastAPI application entry point
- `core/config.py` - Application configuration
- `core/database.py` - Database connections
- `core/redis.py` - Redis caching
- `core/security.py` - Authentication & security
- `models/` - SQLAlchemy database models
- `api/v1/endpoints/` - REST API endpoints
- `services/` - Business logic services
- `schemas/` - Pydantic request/response schemas

### **Key Services**
- `auth.py` - Authentication & user management
- `video_service.py` - Video processing & streaming
- `ml_service.py` - AI/ML recommendations
- `analytics_service.py` - Analytics & metrics
- `storage_service.py` - File storage management
- `notification_service.py` - Notifications
- `live_streaming_service.py` - Live streaming

### **API Endpoints**
- `/auth` - Authentication endpoints
- `/users` - User management
- `/videos` - Video upload & streaming
- `/posts` - Social media posts
- `/live` - Live streaming
- `/analytics` - Analytics & reporting
- `/search` - Search functionality

## 🗄️ **Database Models**
- `User` - User accounts & profiles
- `Video` - Video metadata & processing
- `Post` - Social media posts
- `LiveStream` - Live streaming sessions
- `Comment` - Comments on posts/videos
- `Like` - User likes & reactions
- `Follow` - User relationships
- `Notification` - User notifications
- `Analytics` - Metrics & analytics data

## 🛠️ **Development**

### **Requirements**
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose

### **Setup**
```bash
# Clone repository
git clone https://github.com/nirmal-mina/social-flow-backend.git

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Start services
docker-compose up -d postgres redis

# Run migrations
alembic upgrade head

# Start application
uvicorn app.main:app --reload
```

### **Testing**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific tests
pytest tests/test_auth.py
```

## 🚀 **Deployment**

### **Docker**
```bash
# Build image
docker build -t social-flow-backend .

# Run container
docker run -p 8000:8000 social-flow-backend
```

### **Production**
- **Database**: PostgreSQL with connection pooling
- **Cache**: Redis cluster for high availability
- **Web Server**: Uvicorn with multiple workers
- **Load Balancer**: Nginx or AWS ALB
- **Monitoring**: Prometheus + Grafana

## 🔗 **Key Technologies**

- **Backend**: FastAPI + Python 3.11
- **Database**: PostgreSQL + SQLAlchemy
- **Cache**: Redis
- **Queue**: Celery + Redis
- **Storage**: AWS S3 + CloudFront
- **Video**: AWS MediaConvert + IVS
- **Search**: Elasticsearch
- **Monitoring**: Prometheus + Grafana
- **Testing**: Pytest + Coverage

---

**📝 For detailed documentation, see the full README.md**
