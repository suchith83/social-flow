# Social Flow Backend - Code Fixes and Diffs

**Generated:** 2025-10-06  
**Purpose:** Concrete implementation fixes grouped by subsystem

---

## 1. Health Check Hardening

### File: `app/api/v1/endpoints/health.py`

**Issue:** Health checks may fail completely when optional subsystems unavailable  
**Fix:** Parallel execution with graceful error handling

```python
# BEFORE (Sequential with potential failures)
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    checks = {}
    
    # Check database
    db_healthy, db_info = await check_database(db)
    checks["database"] = db_info
    
    # Check Redis - may throw exception
    redis_healthy, redis_info = await check_redis()
    checks["redis"] = redis_info
    
    # If above fails, these never execute
    s3_healthy, s3_info = await check_s3()
    checks["s3"] = s3_info
    
    return {"status": "healthy" if all_healthy else "unhealthy", "checks": checks}

# AFTER (Parallel with graceful degradation)
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Detailed health check endpoint.
    
    Performs comprehensive health checks on all system components with
    graceful degradation for optional subsystems.
    """
    checks = {}
    
    # Build check list based on feature flags
    tasks = []
    check_names = []
    
    # Database (required)
    tasks.append(check_database(db))
    check_names.append("database")
    
    # Redis (optional)
    if settings.FEATURE_REDIS_ENABLED:
        tasks.append(check_redis())
        check_names.append("redis")
    else:
        checks["redis"] = {"status": "skipped", "reason": "feature disabled"}
    
    # Celery (optional)
    if settings.FEATURE_CELERY_ENABLED:
        tasks.append(check_celery())
        check_names.append("celery")
    else:
        checks["celery"] = {"status": "skipped", "reason": "feature disabled"}
    
    # S3 (optional)
    if settings.FEATURE_S3_ENABLED:
        tasks.append(check_s3())
        check_names.append("s3")
    else:
        checks["s3"] = {"status": "skipped", "reason": "feature disabled"}
    
    # ML Models (optional)
    if settings.FEATURE_ML_ENABLED:
        tasks.append(check_ml_models())
        check_names.append("ml_models")
    else:
        checks["ml_models"] = {"status": "skipped", "reason": "feature disabled"}
    
    # Execute all checks in parallel with exception handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_healthy = True
    for name, result in zip(check_names, results):
        # Handle both exceptions and tuple returns
        if isinstance(result, BaseException):
            checks[name] = {
                "status": "error",
                "error": str(result),
                "type": type(result).__name__
            }
            # Only mark unhealthy if it's a required service
            if name in ["database"]:
                all_healthy = False
        else:
            healthy, info = result
            checks[name] = info
            if not healthy and name in ["database"]:  # Only required services affect overall status
                all_healthy = False
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }
```

**Update check_s3 for better error handling:**

```python
async def check_s3() -> tuple[bool, Dict[str, Any]]:
    """Check S3 storage connectivity."""
    try:
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            return False, {
                "status": "unconfigured",
                "message": "S3 credentials not configured"
            }
        
        start_time = datetime.utcnow()
        
        # Use boto3 async client
        session = get_session()
        async with session.create_client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        ) as client:
            # Test bucket access
            await client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
        
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return True, {
            "status": "healthy",
            "latency_ms": round(latency, 2),
            "bucket": settings.S3_BUCKET_NAME,
            "region": settings.AWS_REGION
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        return False, {
            "status": "error",
            "error": str(e),
            "error_code": error_code
        }
    except Exception as e:
        return False, {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }
```

---

## 2. AI Models Package Stubs

### Create: `app/ai_models/__init__.py`

```python
"""
AI Models Package - Stubs for graceful degradation.

These modules provide fallback implementations when ML models are not available.
"""

from app.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "content_moderation",
    "recommendation",
    "video_analysis",
    "sentiment_analysis",
    "trending_prediction",
]

logger.warning("AI models package loaded with stub implementations")
```

### Create: `app/ai_models/content_moderation.py`

```python
"""Content Moderation Model Stub."""

from typing import Dict, Any

class ContentModerationModel:
    """Stub content moderation model."""
    
    def __init__(self):
        self.loaded = False
    
    def moderate_text(self, text: str) -> Dict[str, Any]:
        """Return safe default moderation result."""
        return {
            "is_safe": True,
            "toxicity_score": 0.0,
            "categories": {},
            "flagged": False,
            "model": "stub"
        }
    
    def moderate_image(self, image_url: str) -> Dict[str, Any]:
        """Return safe default for image moderation."""
        return {
            "is_safe": True,
            "nsfw_score": 0.0,
            "flagged": False,
            "model": "stub"
        }

def load_model():
    """Load stub model."""
    return ContentModerationModel()
```

### Create: `app/ai_models/recommendation.py`

```python
"""Recommendation Model Stub."""

from typing import List, Dict, Any

class RecommendationModel:
    """Stub recommendation model."""
    
    def __init__(self):
        self.loaded = False
    
    def get_recommendations(
        self,
        user_id: str,
        content_type: str = "video",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return empty recommendations (fallback to trending)."""
        return []
    
    def similar_content(self, content_id: str, limit: int = 10) -> List[str]:
        """Return empty similar content."""
        return []

def load_model():
    """Load stub model."""
    return RecommendationModel()
```

### Create: `app/ai_models/video_analysis.py`

```python
"""Video Analysis Model Stub."""

from typing import Dict, Any, List

class VideoAnalysisModel:
    """Stub video analysis model."""
    
    def __init__(self):
        self.loaded = False
    
    def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """Return minimal video analysis."""
        return {
            "scenes": [],
            "objects": [],
            "faces": [],
            "text": [],
            "model": "stub"
        }
    
    def generate_tags(self, video_url: str) -> List[str]:
        """Return empty tags."""
        return []

def load_model():
    """Load stub model."""
    return VideoAnalysisModel()
```

### Create: `app/ai_models/sentiment_analysis.py`

```python
"""Sentiment Analysis Model Stub."""

from typing import Dict, Any

class SentimentAnalysisModel:
    """Stub sentiment analysis model."""
    
    def __init__(self):
        self.loaded = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Return neutral sentiment."""
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "model": "stub"
        }

def load_model():
    """Load stub model."""
    return SentimentAnalysisModel()
```

### Create: `app/ai_models/trending_prediction.py`

```python
"""Trending Prediction Model Stub."""

from typing import List, Dict, Any

class TrendingPredictionModel:
    """Stub trending prediction model."""
    
    def __init__(self):
        self.loaded = False
    
    def predict_trending(
        self,
        content_type: str = "video",
        timeframe: str = "24h"
    ) -> List[Dict[str, Any]]:
        """Return empty predictions (fallback to view count)."""
        return []

def load_model():
    """Load stub model."""
    return TrendingPredictionModel()
```

---

## 3. User Model is_superuser Property

### File: `app/models/user.py`

**Issue:** Docs/old code may expect `is_superuser` boolean  
**Fix:** Add compatibility property (ALREADY IMPLEMENTED - VERIFY WORKS)

```python
# In User model class
@property
def is_superuser(self) -> bool:
    """
    Compatibility property for superuser checks.
    Returns True if user has admin or super_admin role.
    """
    return self.role in (UserRole.ADMIN, UserRole.SUPER_ADMIN)
```

**Verify this property exists and test it:**

```python
# Test file: tests/unit/test_user_model.py
import pytest
from app.models.user import User, UserRole

def test_is_superuser_property():
    """Test is_superuser compatibility property."""
    # Admin user
    admin = User(
        username="admin",
        email="admin@test.com",
        role=UserRole.ADMIN
    )
    assert admin.is_superuser is True
    
    # Super admin
    super_admin = User(
        username="superadmin",
        email="super@test.com",
        role=UserRole.SUPER_ADMIN
    )
    assert super_admin.is_superuser is True
    
    # Regular user
    user = User(
        username="user",
        email="user@test.com",
        role=UserRole.USER
    )
    assert user.is_superuser is False
    
    # Creator
    creator = User(
        username="creator",
        email="creator@test.com",
        role=UserRole.CREATOR
    )
    assert creator.is_superuser is False
```

---

## 4. Test Conversion Example

### Convert: `comprehensive_test.py` → `tests/integration/test_comprehensive.py`

```python
"""
Comprehensive Integration Tests.

Converted from comprehensive_test.py report-style script to real pytest suite.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient

@pytest.mark.asyncio
class TestCoreImports:
    """Test that all core modules can be imported."""
    
    async def test_main_app_import(self):
        """Test main application import."""
        from app.main import app
        assert app is not None
        assert app.title == "Social Flow Backend"
    
    async def test_config_import(self):
        """Test configuration import."""
        from app.core.config import settings
        assert settings.PROJECT_NAME == "Social Flow Backend"
        assert settings.API_V1_STR == "/api/v1"
    
    async def test_database_import(self):
        """Test database base import."""
        from app.core.database import Base
        assert Base is not None
    
    async def test_models_import(self):
        """Test all model imports."""
        from app.models.user import User
        from app.models.video import Video
        from app.models.social import Post, Comment, Like
        from app.models.payment import Payment, Subscription
        
        assert User.__tablename__ == "users"
        assert Video.__tablename__ == "videos"
        assert Post.__tablename__ == "posts"
    
    async def test_schemas_import(self):
        """Test schema imports."""
        from app.schemas import user, video, social
        assert user is not None
        assert video is not None
        assert social is not None
    
    async def test_services_import(self):
        """Test service layer imports."""
        from app.services.storage_service import StorageService
        from app.services.recommendation_service import RecommendationService
        from app.services.search_service import SearchService
        
        assert StorageService is not None
        assert RecommendationService is not None
        assert SearchService is not None


@pytest.mark.asyncio
class TestConfiguration:
    """Test configuration management."""
    
    async def test_required_settings(self):
        """Test required settings are present."""
        from app.core.config import settings
        
        assert settings.PROJECT_NAME
        assert settings.VERSION
        assert settings.API_V1_STR
        assert settings.SECRET_KEY
        assert settings.ALGORITHM
    
    async def test_database_url(self):
        """Test database URL configuration."""
        from app.core.config import settings
        
        assert settings.DATABASE_URL
        assert "://" in settings.DATABASE_URL
    
    async def test_feature_flags(self):
        """Test feature flags are accessible."""
        from app.core.config import settings
        
        assert hasattr(settings, 'FEATURE_S3_ENABLED')
        assert hasattr(settings, 'FEATURE_REDIS_ENABLED')
        assert hasattr(settings, 'FEATURE_ML_ENABLED')


@pytest.mark.asyncio
class TestDatabaseModels:
    """Test database model instantiation."""
    
    async def test_user_model_creation(self, db_session: AsyncSession):
        """Test user model can be created."""
        from app.models.user import User, UserRole, UserStatus
        
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed",
            role=UserRole.USER,
            status=UserStatus.ACTIVE
        )
        
        assert user.username == "testuser"
        assert user.role == UserRole.USER
    
    async def test_video_model_creation(self, db_session: AsyncSession):
        """Test video model can be created."""
        from app.models.video import Video
        from app.models.user import User
        import uuid
        
        user_id = uuid.uuid4()
        video = Video(
            title="Test Video",
            description="Test Description",
            user_id=user_id,
            file_key="test_key"
        )
        
        assert video.title == "Test Video"
        assert video.user_id == user_id
```

---

## 5. Pydantic V2 Migration

### File: `app/auth/schemas/auth.py`

```python
# BEFORE (Pydantic V1)
from pydantic import BaseModel, validator

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v
    
    class Config:
        schema_extra = {"example": {...}}

# AFTER (Pydantic V2)
from pydantic import BaseModel, field_validator, ConfigDict

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        assert v.isalnum(), 'must be alphanumeric'
        return v
    
    model_config = ConfigDict(
        json_schema_extra={"example": {...}}
    )
```

### File: `app/posts/schemas/post.py`

```python
# BEFORE
from pydantic import validator

class PostCreate(BaseModel):
    content: str
    original_post_id: Optional[UUID] = None
    
    @validator('original_post_id')
    def validate_repost(cls, v, values):
        if v and not values.get('content'):
            raise ValueError('Repost must have content')
        return v

# AFTER
from pydantic import field_validator

class PostCreate(BaseModel):
    content: str
    original_post_id: Optional[UUID] = None
    
    @field_validator('original_post_id')
    @classmethod
    def validate_repost(cls, v: Optional[UUID], info) -> Optional[UUID]:
        # In Pydantic V2, use info.data to access other fields
        if v and not info.data.get('content'):
            raise ValueError('Repost must have content')
        return v
```

---

## 6. Router Cleanup

### File: `app/api/v1/router.py`

```python
# BEFORE (commented duplicates)
# from app.notifications.api import notifications  # removed duplicate
from app.analytics.api import analytics  # canonical analytics router
# from app.analytics.routes import analytics_routes as analytics_enhanced  # removed duplicate

# AFTER (clean, with comments explaining decisions)
# Notification endpoint (single canonical source)
from app.api.v1.endpoints import notifications as notifications_endpoints

# Analytics endpoint (consolidated into single router)
from app.analytics.api import analytics

# ... in router registration ...

# Notifications (v1 endpoints are canonical)
api_router.include_router(
    notifications_endpoints.router,
    prefix="/notifications",
    tags=["notifications"]
)

# Analytics (canonical aggregated analytics)
api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["analytics"]
)
```

---

## 7. Environment Configuration

### File: `env.example`

```bash
# BEFORE (incomplete)
SECRET_KEY=your-secret-key
POSTGRES_USER=postgres

# AFTER (complete with all required vars)
# === Application ===
PROJECT_NAME="Social Flow Backend"
VERSION="1.0.0"
ENVIRONMENT=development
DEBUG=True
TESTING=False

# === Security ===
SECRET_KEY=your-secret-key-generate-with-openssl-rand-hex-32
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# === Database ===
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DB=social_flow
POSTGRES_PORT=5432
DATABASE_URL=postgresql+asyncpg://postgres:your-password@localhost:5432/social_flow

# === Redis ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_URL=redis://localhost:6379/0

# === AWS / S3 ===
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=social-flow-videos
AWS_CLOUDFRONT_DOMAIN=

# === Stripe ===
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# === Feature Flags ===
FEATURE_S3_ENABLED=True
FEATURE_REDIS_ENABLED=True
FEATURE_ML_ENABLED=True
FEATURE_CELERY_ENABLED=True

# === OAuth ===
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
OAUTH_REDIRECT_URI=http://localhost:3000/auth/callback

# === Email ===
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
EMAIL_FROM=

# === Frontend ===
FRONTEND_URL=http://localhost:3000
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# === Monitoring ===
ENABLE_METRICS=True
LOG_LEVEL=INFO
SENTRY_DSN=
```

---

## Summary of Code Changes

| File | Issue | Lines Changed | Complexity |
|------|-------|---------------|------------|
| app/api/v1/endpoints/health.py | CRIT-005 | ~100 | Medium |
| app/ai_models/*.py | CRIT-001 | ~300 | Low (stubs) |
| app/models/user.py | CONS-007 | ~5 | Low |
| tests/integration/test_comprehensive.py | TEST-001 | ~200 | Medium |
| app/auth/schemas/auth.py | CONS-001 | ~20 | Low |
| app/posts/schemas/post.py | CONS-001 | ~15 | Low |
| app/api/v1/router.py | CONS-003 | ~10 | Low |
| env.example | DOC-004 | ~50 | Low |

**Total Estimated LOC:** ~700 lines
**Total Files Modified:** 8-15 files
**New Files Created:** 6-10 files

---

**See test_strategy.md for test implementation details**
