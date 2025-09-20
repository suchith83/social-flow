# Testing Guide

This guide covers comprehensive testing strategies for the Social Flow Backend, including unit tests, integration tests, end-to-end tests, and performance testing.

## 🧪 Testing Strategy

### Testing Pyramid

```
        /\
       /  \
      / E2E \     End-to-End Tests (5%)
     /______\
    /        \
   /Integration\  Integration Tests (25%)
  /____________\
 /              \
/   Unit Tests   \  Unit Tests (70%)
/________________\
```

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test API endpoints and database interactions
3. **End-to-End Tests**: Test complete user workflows
4. **Performance Tests**: Test system performance under load
5. **Security Tests**: Test security vulnerabilities
6. **Contract Tests**: Test API contracts

## 🛠️ Testing Setup

### Dependencies

Add testing dependencies to `requirements.txt`:

```txt
# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
pytest-xdist==3.3.1
httpx==0.25.2
factory-boy==3.3.0
faker==20.1.0
freezegun==1.2.2
responses==0.23.3
```

### Test Configuration

Create `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=app
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    --asyncio-mode=auto
    -v
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
```

### Test Database Setup

Create `tests/conftest.py`:

```python
import asyncio
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db, Base
from app.core.config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.fixture
async def test_db(test_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session

@pytest.fixture
def client(test_db):
    """Create test client."""
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
async def async_client(test_db):
    """Create async test client."""
    def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()

@pytest.fixture
def sample_user():
    """Create sample user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "display_name": "Test User",
        "bio": "Test bio",
    }

@pytest.fixture
def sample_video():
    """Create sample video data."""
    return {
        "title": "Test Video",
        "description": "Test video description",
        "tags": "test,video",
        "filename": "test_video.mp4",
        "file_size": 1024000,
        "duration": 120.5,
        "resolution": "1920x1080",
    }
```

## 🔬 Unit Tests

### Authentication Tests

```python
# tests/unit/test_auth.py
import pytest
from unittest.mock import Mock, patch
from app.services.auth import AuthService
from app.core.security import verify_password, get_password_hash
from app.models.user import User

class TestAuthService:
    def test_verify_password(self):
        """Test password verification."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)
    
    @pytest.mark.asyncio
    async def test_create_user(self, test_db, sample_user):
        """Test user creation."""
        auth_service = AuthService(test_db)
        
        user = await auth_service.create_user(sample_user)
        
        assert user.username == sample_user["username"]
        assert user.email == sample_user["email"]
        assert user.display_name == sample_user["display_name"]
        assert verify_password(sample_user["password"], user.hashed_password)
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, test_db, sample_user):
        """Test user authentication."""
        auth_service = AuthService(test_db)
        
        # Create user
        await auth_service.create_user(sample_user)
        
        # Authenticate user
        user = await auth_service.authenticate_user(
            sample_user["username"], 
            sample_user["password"]
        )
        
        assert user is not None
        assert user.username == sample_user["username"]
    
    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, test_db, sample_user):
        """Test user authentication with wrong password."""
        auth_service = AuthService(test_db)
        
        # Create user
        await auth_service.create_user(sample_user)
        
        # Try to authenticate with wrong password
        user = await auth_service.authenticate_user(
            sample_user["username"], 
            "wrongpassword"
        )
        
        assert user is None
    
    @pytest.mark.asyncio
    async def test_get_user_by_username(self, test_db, sample_user):
        """Test getting user by username."""
        auth_service = AuthService(test_db)
        
        # Create user
        created_user = await auth_service.create_user(sample_user)
        
        # Get user by username
        user = await auth_service.get_user_by_username(sample_user["username"])
        
        assert user is not None
        assert user.id == created_user.id
        assert user.username == sample_user["username"]
    
    @pytest.mark.asyncio
    async def test_get_user_by_email(self, test_db, sample_user):
        """Test getting user by email."""
        auth_service = AuthService(test_db)
        
        # Create user
        created_user = await auth_service.create_user(sample_user)
        
        # Get user by email
        user = await auth_service.get_user_by_email(sample_user["email"])
        
        assert user is not None
        assert user.id == created_user.id
        assert user.email == sample_user["email"]
```

### Video Service Tests

```python
# tests/unit/test_video_service.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.video_service import VideoService
from app.models.video import Video
from app.models.user import User

class TestVideoService:
    @pytest.mark.asyncio
    async def test_upload_video(self, test_db, sample_user, sample_video):
        """Test video upload."""
        video_service = VideoService(test_db)
        
        # Create user
        user = User(
            username=sample_user["username"],
            email=sample_user["email"],
            hashed_password="hashed_password",
            display_name=sample_user["display_name"]
        )
        test_db.add(user)
        await test_db.commit()
        
        # Mock file upload
        mock_file = Mock()
        mock_file.filename = sample_video["filename"]
        mock_file.size = sample_video["file_size"]
        
        with patch('app.services.video_service.upload_to_s3') as mock_s3:
            mock_s3.return_value = "https://s3.amazonaws.com/bucket/video.mp4"
            
            result = await video_service.upload_video(
                mock_file, user, sample_video
            )
            
            assert result["video_id"] is not None
            assert result["status"] == "uploaded"
            mock_s3.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_video_by_id(self, test_db, sample_user, sample_video):
        """Test getting video by ID."""
        video_service = VideoService(test_db)
        
        # Create user and video
        user = User(
            username=sample_user["username"],
            email=sample_user["email"],
            hashed_password="hashed_password",
            display_name=sample_user["display_name"]
        )
        test_db.add(user)
        await test_db.commit()
        
        video = Video(
            title=sample_video["title"],
            description=sample_video["description"],
            filename=sample_video["filename"],
            owner_id=user.id
        )
        test_db.add(video)
        await test_db.commit()
        
        # Get video
        result = await video_service.get_video_by_id(str(video.id))
        
        assert result is not None
        assert result.title == sample_video["title"]
        assert result.owner_id == user.id
    
    @pytest.mark.asyncio
    async def test_like_video(self, test_db, sample_user, sample_video):
        """Test liking a video."""
        video_service = VideoService(test_db)
        
        # Create user and video
        user = User(
            username=sample_user["username"],
            email=sample_user["email"],
            hashed_password="hashed_password",
            display_name=sample_user["display_name"]
        )
        test_db.add(user)
        await test_db.commit()
        
        video = Video(
            title=sample_video["title"],
            filename=sample_video["filename"],
            owner_id=user.id
        )
        test_db.add(video)
        await test_db.commit()
        
        # Like video
        result = await video_service.like_video(str(video.id), user.id)
        
        assert result["liked"] is True
        assert result["likes_count"] == 1
    
    @pytest.mark.asyncio
    async def test_increment_view_count(self, test_db, sample_user, sample_video):
        """Test incrementing view count."""
        video_service = VideoService(test_db)
        
        # Create user and video
        user = User(
            username=sample_user["username"],
            email=sample_user["email"],
            hashed_password="hashed_password",
            display_name=sample_user["display_name"]
        )
        test_db.add(user)
        await test_db.commit()
        
        video = Video(
            title=sample_video["title"],
            filename=sample_video["filename"],
            owner_id=user.id
        )
        test_db.add(video)
        await test_db.commit()
        
        # Increment view count
        result = await video_service.increment_view_count(str(video.id), str(user.id))
        
        assert result["views_count"] == 1
```

### ML Service Tests

```python
# tests/unit/test_ml_service.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.ml_service import MLService

class TestMLService:
    @pytest.mark.asyncio
    async def test_analyze_content(self, test_db):
        """Test content analysis."""
        ml_service = MLService(test_db)
        
        content_data = {
            "text": "This is a test message",
            "user_id": "123"
        }
        
        with patch('app.services.ml_service.sagemaker_client') as mock_sagemaker:
            mock_sagemaker.invoke_endpoint.return_value = {
                'Body': Mock(read=lambda: b'{"sentiment": "positive", "toxicity": 0.1}')
            }
            
            result = await ml_service.analyze_content("text", content_data)
            
            assert result["task_id"] is not None
            assert result["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_get_recommendations(self, test_db):
        """Test getting recommendations."""
        ml_service = MLService(test_db)
        
        with patch('app.services.ml_service.sagemaker_client') as mock_sagemaker:
            mock_sagemaker.invoke_endpoint.return_value = {
                'Body': Mock(read=lambda: b'{"recommendations": ["video1", "video2", "video3"]}')
            }
            
            result = await ml_service.get_recommendations("mixed", 10)
            
            assert result["task_id"] is not None
            assert result["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, test_db):
        """Test getting task status."""
        ml_service = MLService(test_db)
        
        with patch('app.services.ml_service.celery_app') as mock_celery:
            mock_task = Mock()
            mock_task.state = "SUCCESS"
            mock_task.result = {"sentiment": "positive", "toxicity": 0.1}
            mock_celery.AsyncResult.return_value = mock_task
            
            result = await ml_service.get_task_status("task123")
            
            assert result["status"] == "completed"
            assert result["result"]["sentiment"] == "positive"
```

## 🔗 Integration Tests

### API Endpoint Tests

```python
# tests/integration/test_api_auth.py
import pytest
from httpx import AsyncClient
from app.main import app

class TestAuthAPI:
    @pytest.mark.asyncio
    async def test_register_user(self, async_client: AsyncClient, sample_user):
        """Test user registration endpoint."""
        response = await async_client.post("/auth/register", json=sample_user)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == sample_user["username"]
        assert data["email"] == sample_user["email"]
        assert "id" in data
        assert "created_at" in data
    
    @pytest.mark.asyncio
    async def test_register_duplicate_user(self, async_client: AsyncClient, sample_user):
        """Test registering duplicate user."""
        # First registration
        await async_client.post("/auth/register", json=sample_user)
        
        # Second registration with same username
        response = await async_client.post("/auth/register", json=sample_user)
        
        assert response.status_code == 409
    
    @pytest.mark.asyncio
    async def test_login_user(self, async_client: AsyncClient, sample_user):
        """Test user login endpoint."""
        # Register user first
        await async_client.post("/auth/register", json=sample_user)
        
        # Login user
        response = await async_client.post(
            "/auth/login",
            data={
                "username": sample_user["username"],
                "password": sample_user["password"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, async_client: AsyncClient, sample_user):
        """Test login with invalid credentials."""
        # Register user first
        await async_client.post("/auth/register", json=sample_user)
        
        # Login with wrong password
        response = await async_client.post(
            "/auth/login",
            data={
                "username": sample_user["username"],
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: AsyncClient, sample_user):
        """Test token refresh endpoint."""
        # Register and login user
        await async_client.post("/auth/register", json=sample_user)
        login_response = await async_client.post(
            "/auth/login",
            data={
                "username": sample_user["username"],
                "password": sample_user["password"]
            }
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh token
        response = await async_client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
```

### Video API Tests

```python
# tests/integration/test_api_videos.py
import pytest
from httpx import AsyncClient
from app.main import app

class TestVideoAPI:
    @pytest.fixture
    async def authenticated_client(self, async_client: AsyncClient, sample_user):
        """Create authenticated client."""
        # Register and login user
        await async_client.post("/auth/register", json=sample_user)
        login_response = await async_client.post(
            "/auth/login",
            data={
                "username": sample_user["username"],
                "password": sample_user["password"]
            }
        )
        token = login_response.json()["access_token"]
        
        # Set authorization header
        async_client.headers.update({"Authorization": f"Bearer {token}"})
        return async_client
    
    @pytest.mark.asyncio
    async def test_upload_video(self, authenticated_client: AsyncClient):
        """Test video upload endpoint."""
        # Create test video file
        test_file = b"fake video content"
        
        response = await authenticated_client.post(
            "/videos/upload",
            files={"file": ("test_video.mp4", test_file, "video/mp4")},
            data={
                "title": "Test Video",
                "description": "Test video description",
                "tags": "test,video"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "video_id" in data
        assert data["status"] == "uploaded"
    
    @pytest.mark.asyncio
    async def test_get_video(self, authenticated_client: AsyncClient):
        """Test get video endpoint."""
        # Upload video first
        test_file = b"fake video content"
        upload_response = await authenticated_client.post(
            "/videos/upload",
            files={"file": ("test_video.mp4", test_file, "video/mp4")},
            data={"title": "Test Video"}
        )
        video_id = upload_response.json()["video_id"]
        
        # Get video
        response = await authenticated_client.get(f"/videos/{video_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Video"
        assert data["id"] == video_id
    
    @pytest.mark.asyncio
    async def test_like_video(self, authenticated_client: AsyncClient):
        """Test like video endpoint."""
        # Upload video first
        test_file = b"fake video content"
        upload_response = await authenticated_client.post(
            "/videos/upload",
            files={"file": ("test_video.mp4", test_file, "video/mp4")},
            data={"title": "Test Video"}
        )
        video_id = upload_response.json()["video_id"]
        
        # Like video
        response = await authenticated_client.post(f"/videos/{video_id}/like")
        
        assert response.status_code == 200
        data = response.json()
        assert data["liked"] is True
        assert data["likes_count"] == 1
    
    @pytest.mark.asyncio
    async def test_record_view(self, authenticated_client: AsyncClient):
        """Test record view endpoint."""
        # Upload video first
        test_file = b"fake video content"
        upload_response = await authenticated_client.post(
            "/videos/upload",
            files={"file": ("test_video.mp4", test_file, "video/mp4")},
            data={"title": "Test Video"}
        )
        video_id = upload_response.json()["video_id"]
        
        # Record view
        response = await authenticated_client.post(f"/videos/{video_id}/view")
        
        assert response.status_code == 200
        data = response.json()
        assert data["views_count"] == 1
```

## 🎯 End-to-End Tests

### Complete User Workflow Tests

```python
# tests/e2e/test_user_workflow.py
import pytest
from httpx import AsyncClient
from app.main import app

class TestUserWorkflow:
    @pytest.mark.asyncio
    async def test_complete_user_journey(self, async_client: AsyncClient):
        """Test complete user journey from registration to video upload."""
        # 1. Register user
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123",
            "display_name": "Test User"
        }
        
        register_response = await async_client.post("/auth/register", json=user_data)
        assert register_response.status_code == 201
        
        # 2. Login user
        login_response = await async_client.post(
            "/auth/login",
            data={
                "username": user_data["username"],
                "password": user_data["password"]
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # 3. Set authorization header
        async_client.headers.update({"Authorization": f"Bearer {token}"})
        
        # 4. Upload video
        test_file = b"fake video content"
        upload_response = await async_client.post(
            "/videos/upload",
            files={"file": ("test_video.mp4", test_file, "video/mp4")},
            data={
                "title": "My First Video",
                "description": "This is my first video upload",
                "tags": "first,video,test"
            }
        )
        assert upload_response.status_code == 200
        video_id = upload_response.json()["video_id"]
        
        # 5. Get video details
        video_response = await async_client.get(f"/videos/{video_id}")
        assert video_response.status_code == 200
        video_data = video_response.json()
        assert video_data["title"] == "My First Video"
        
        # 6. Like video
        like_response = await async_client.post(f"/videos/{video_id}/like")
        assert like_response.status_code == 200
        
        # 7. Record view
        view_response = await async_client.post(f"/videos/{video_id}/view")
        assert view_response.status_code == 200
        
        # 8. Get updated video details
        updated_video_response = await async_client.get(f"/videos/{video_id}")
        assert updated_video_response.status_code == 200
        updated_video_data = updated_video_response.json()
        assert updated_video_data["likes_count"] == 1
        assert updated_video_data["views_count"] == 1
    
    @pytest.mark.asyncio
    async def test_live_streaming_workflow(self, async_client: AsyncClient):
        """Test live streaming workflow."""
        # 1. Register and login user
        user_data = {
            "username": "streamer",
            "email": "streamer@example.com",
            "password": "testpassword123",
            "display_name": "Streamer"
        }
        
        await async_client.post("/auth/register", json=user_data)
        login_response = await async_client.post(
            "/auth/login",
            data={
                "username": user_data["username"],
                "password": user_data["password"]
            }
        )
        token = login_response.json()["access_token"]
        async_client.headers.update({"Authorization": f"Bearer {token}"})
        
        # 2. Create live stream
        stream_response = await async_client.post(
            "/videos/live/create",
            data={
                "title": "My Live Stream",
                "description": "Live streaming test"
            }
        )
        assert stream_response.status_code == 200
        stream_data = stream_response.json()
        assert "stream_id" in stream_data
        assert "stream_key" in stream_data
        assert "rtmp_url" in stream_data
        
        # 3. End live stream
        end_stream_response = await async_client.post(
            f"/videos/live/{stream_data['stream_id']}/end"
        )
        assert end_stream_response.status_code == 200
```

## ⚡ Performance Tests

### Load Testing with Locust

Create `tests/performance/locustfile.py`:

```python
from locust import HttpUser, task, between
import random
import string

class SocialFlowUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts."""
        self.username = self.generate_username()
        self.email = f"{self.username}@example.com"
        self.password = "testpassword123"
        
        # Register user
        response = self.client.post("/auth/register", json={
            "username": self.username,
            "email": self.email,
            "password": self.password,
            "display_name": f"User {self.username}"
        })
        
        if response.status_code == 201:
            # Login user
            login_response = self.client.post("/auth/login", data={
                "username": self.username,
                "password": self.password
            })
            if login_response.status_code == 200:
                self.token = login_response.json()["access_token"]
                self.client.headers.update({
                    "Authorization": f"Bearer {self.token}"
                })
    
    def generate_username(self):
        """Generate random username."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    @task(3)
    def get_videos(self):
        """Get videos list."""
        self.client.get("/videos")
    
    @task(2)
    def get_user_profile(self):
        """Get user profile."""
        self.client.get("/users/me")
    
    @task(1)
    def upload_video(self):
        """Upload video."""
        test_file = b"fake video content"
        self.client.post(
            "/videos/upload",
            files={"file": ("test_video.mp4", test_file, "video/mp4")},
            data={"title": f"Test Video {random.randint(1, 1000)}"}
        )
    
    @task(2)
    def like_video(self):
        """Like a random video."""
        # Get videos first
        videos_response = self.client.get("/videos?limit=10")
        if videos_response.status_code == 200:
            videos = videos_response.json()
            if videos:
                video_id = random.choice(videos)["id"]
                self.client.post(f"/videos/{video_id}/like")
    
    @task(1)
    def get_analytics(self):
        """Get analytics data."""
        self.client.get("/analytics/user/me")
```

### Performance Test Configuration

Create `tests/performance/performance_test.py`:

```python
import pytest
import asyncio
import time
from httpx import AsyncClient
from app.main import app

class TestPerformance:
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_requests(self, async_client: AsyncClient):
        """Test concurrent request handling."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(100):
            task = async_client.get("/health")
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Check all requests succeeded
        assert all(response.status_code == 200 for response in responses)
        
        # Check response time
        response_time = end_time - start_time
        assert response_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_database_performance(self, async_client: AsyncClient):
        """Test database query performance."""
        # Create test data
        for i in range(1000):
            await async_client.post("/auth/register", json={
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "password": "testpassword123",
                "display_name": f"User {i}"
            })
        
        # Test query performance
        start_time = time.time()
        response = await async_client.get("/users?limit=100")
        end_time = time.time()
        
        assert response.status_code == 200
        query_time = end_time - start_time
        assert query_time < 1.0  # Should complete within 1 second
```

## 🔒 Security Tests

### Security Test Suite

```python
# tests/security/test_security.py
import pytest
from httpx import AsyncClient
from app.main import app

class TestSecurity:
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_sql_injection(self, async_client: AsyncClient):
        """Test SQL injection protection."""
        # Try SQL injection in username field
        malicious_username = "admin'; DROP TABLE users; --"
        
        response = await async_client.post("/auth/register", json={
            "username": malicious_username,
            "email": "test@example.com",
            "password": "testpassword123",
            "display_name": "Test User"
        })
        
        # Should return validation error, not execute SQL
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_xss_protection(self, async_client: AsyncClient):
        """Test XSS protection."""
        # Try XSS in display name
        xss_payload = "<script>alert('XSS')</script>"
        
        response = await async_client.post("/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123",
            "display_name": xss_payload
        })
        
        # Should sanitize input
        assert response.status_code == 201
        data = response.json()
        assert "<script>" not in data["display_name"]
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_rate_limiting(self, async_client: AsyncClient):
        """Test rate limiting."""
        # Make multiple requests quickly
        for i in range(10):
            response = await async_client.post("/auth/login", data={
                "username": "nonexistent",
                "password": "wrongpassword"
            })
        
        # Should be rate limited
        assert response.status_code == 429
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_authentication_required(self, async_client: AsyncClient):
        """Test that protected endpoints require authentication."""
        # Try to access protected endpoint without token
        response = await async_client.get("/users/me")
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_invalid_token(self, async_client: AsyncClient):
        """Test invalid token handling."""
        # Try to access protected endpoint with invalid token
        async_client.headers.update({"Authorization": "Bearer invalid_token"})
        response = await async_client.get("/users/me")
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_cors_headers(self, async_client: AsyncClient):
        """Test CORS headers."""
        response = await async_client.options("/auth/register")
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
```

## 📊 Test Coverage

### Coverage Configuration

Create `.coveragerc`:

```ini
[run]
source = app
omit = 
    */tests/*
    */migrations/*
    */alembic/*
    */venv/*
    */env/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
```

### Coverage Commands

```bash
# Run tests with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing

# Generate coverage report
coverage html
coverage xml

# Check coverage threshold
pytest --cov=app --cov-fail-under=80
```

## 🚀 Running Tests

### Test Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m performance
pytest -m security

# Run tests in parallel
pytest -n auto

# Run tests with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_auth.py

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x

# Run tests with debugging
pytest --pdb
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=app --cov-report=xml --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 📈 Test Metrics

### Key Metrics to Track

1. **Test Coverage**: Percentage of code covered by tests
2. **Test Execution Time**: How long tests take to run
3. **Test Success Rate**: Percentage of tests passing
4. **Flaky Test Rate**: Percentage of tests that fail intermittently
5. **Test Maintenance Cost**: Time spent maintaining tests

### Monitoring Test Health

```python
# tests/conftest.py
import pytest
import time
from datetime import datetime

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test execution time and results."""
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call":
        # Log test execution time
        if hasattr(item, 'start_time'):
            execution_time = time.time() - item.start_time
            print(f"Test {item.name} took {execution_time:.2f} seconds")
        
        # Log test result
        if rep.failed:
            print(f"Test {item.name} FAILED: {rep.longrepr}")
        elif rep.passed:
            print(f"Test {item.name} PASSED")

@pytest.fixture(autouse=True)
def track_test_time(request):
    """Track test execution time."""
    request.node.start_time = time.time()
    yield
```

## 🎯 Best Practices

### Test Organization

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One Assertion Per Test**: Keep tests focused
3. **Descriptive Test Names**: Use clear, descriptive names
4. **Test Data Factories**: Use factories for test data
5. **Mock External Dependencies**: Mock external services

### Test Maintenance

1. **Regular Test Reviews**: Review tests regularly
2. **Remove Obsolete Tests**: Remove tests for removed features
3. **Update Test Data**: Keep test data current
4. **Monitor Test Performance**: Track test execution time
5. **Document Test Strategy**: Document testing approach

### Test Quality

1. **High Coverage**: Aim for 80%+ coverage
2. **Fast Execution**: Keep tests fast
3. **Reliable**: Tests should be deterministic
4. **Maintainable**: Easy to understand and modify
5. **Comprehensive**: Cover all critical paths

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Issues**
   - Check test database configuration
   - Ensure database is running
   - Verify connection strings

2. **Async Test Issues**
   - Use `pytest-asyncio`
   - Properly await async functions
   - Check event loop configuration

3. **Mock Issues**
   - Verify mock setup
   - Check mock return values
   - Ensure mocks are called

4. **Test Data Issues**
   - Use factories for test data
   - Clean up test data
   - Avoid hardcoded values

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pytest debugging
pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb

# Print debug information
print(f"Debug: {variable}")

# Use breakpoints
import pdb; pdb.set_trace()
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites)
- [Async Testing](https://pytest-asyncio.readthedocs.io/)
- [Test Coverage](https://coverage.readthedocs.io/)
