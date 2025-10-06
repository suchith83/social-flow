# Social Flow Backend - Test Strategy

**Generated:** 2025-10-06  
**Testing Framework:** pytest, pytest-asyncio, httpx  
**Target Coverage:** 80% overall, 95% critical paths

---

## Test Pyramid

```
        /\
       /E2E\      2-5% of tests (Critical user journeys)
      /------\
     /INTEGR.\ 15-20% (Service + DB + External APIs with mocks)
    /----------\
   /    UNIT    \ 75-80% (Pure business logic, no dependencies)
  /--------------\
```

---

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose:** Test pure business logic without external dependencies

**Characteristics:**
- No database, no Redis, no HTTP clients
- Fast execution (<1ms per test)
- High coverage target: >90%
- Mocked dependencies

**Examples:**

```python
# tests/unit/test_user_model.py
import pytest
from app.models.user import User, UserRole, UserStatus

def test_user_can_post_content_active():
    """Test that active user can post content."""
    user = User(
        username="testuser",
        email="test@example.com",
        status=UserStatus.ACTIVE,
        role=UserRole.USER
    )
    assert user.can_post_content() is True

def test_user_cannot_post_when_banned():
    """Test that banned user cannot post."""
    from datetime import datetime
    user = User(
        username="testuser",
        email="test@example.com",
        status=UserStatus.BANNED,
        banned_at=datetime.utcnow()
    )
    assert user.can_post_content() is False

def test_is_superuser_property():
    """Test is_superuser compatibility property."""
    admin = User(username="admin", email="admin@test.com", role=UserRole.ADMIN)
    assert admin.is_superuser is True
    
    user = User(username="user", email="user@test.com", role=UserRole.USER)
    assert user.is_superuser is False

def test_creator_can_monetize():
    """Test creator monetization check."""
    creator = User(
        username="creator",
        email="creator@test.com",
        is_creator=True,
        stripe_connect_onboarded=True
    )
    assert creator.can_monetize() is True
    
    user = User(username="user", email="user@test.com", is_creator=False)
    assert user.can_monetize() is False
```

```python
# tests/unit/test_video_model.py
import pytest
from app.models.video import Video, VideoStatus, VideoVisibility
from app.models.user import User
import uuid

def test_video_increment_view_count():
    """Test video view count increment."""
    video = Video(
        title="Test",
        user_id=uuid.uuid4(),
        views_count=100
    )
    video.increment_views()
    assert video.views_count == 101

def test_video_is_published():
    """Test video published status check."""
    video = Video(
        title="Test",
        user_id=uuid.uuid4(),
        status=VideoStatus.PUBLISHED
    )
    assert video.is_published() is True

def test_video_visibility_check():
    """Test video visibility logic."""
    video = Video(
        title="Test",
        user_id=uuid.uuid4(),
        visibility=VideoVisibility.PUBLIC
    )
    assert video.is_visible_to(None) is True  # Public to all
    
    private_video = Video(
        title="Test",
        user_id=uuid.uuid4(),
        visibility=VideoVisibility.PRIVATE
    )
    assert private_video.is_visible_to(None) is False  # Not visible to unauthenticated
```

---

### 2. Integration Tests (`tests/integration/`)

**Purpose:** Test service layer with real database and mocked external services

**Characteristics:**
- Uses test database (transaction rollback per test)
- Redis connection (test DB or mock)
- Mocked AWS S3, Stripe, ML models
- Moderate execution time (10-100ms per test)
- Coverage target: >85%

**Examples:**

```python
# tests/integration/test_auth_service.py
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.mark.asyncio
async def test_register_user(async_client: AsyncClient, db_session: AsyncSession):
    """Test user registration flow."""
    response = await async_client.post("/api/v1/auth/register", json={
        "username": "newuser",
        "email": "newuser@test.com",
        "password": "TestPass123!",
        "full_name": "New User"
    })
    
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "newuser@test.com"
    assert "id" in data
    assert "password" not in data  # Never return password

@pytest.mark.asyncio
async def test_login_flow(async_client: AsyncClient, test_user: User):
    """Test login with valid credentials."""
    response = await async_client.post("/api/v1/auth/login", data={
        "username": test_user.email,
        "password": "TestPass123!"  # From fixture
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert isinstance(data["expires_in"], int)

@pytest.mark.asyncio
async def test_protected_endpoint_requires_auth(async_client: AsyncClient):
    """Test that protected endpoint requires authentication."""
    response = await async_client.get("/api/v1/auth/me")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_protected_endpoint_with_valid_token(
    async_client: AsyncClient,
    auth_headers: dict
):
    """Test protected endpoint with valid JWT."""
    response = await async_client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "email" in data
```

```python
# tests/integration/test_video_service.py
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
from app.models.user import User

@pytest.mark.asyncio
async def test_video_upload_initiation(
    async_client: AsyncClient,
    auth_headers: dict,
    mock_s3
):
    """Test video upload initiation with S3 presigned URL."""
    with patch('app.services.storage_service.StorageService.generate_presigned_upload_url') as mock_upload:
        mock_upload.return_value = {
            "url": "https://s3.amazonaws.com/test/upload?signature=abc",
            "fields": {},
            "key": "videos/test-key.mp4"
        }
        
        response = await async_client.post(
            "/api/v1/videos",
            headers=auth_headers,
            json={
                "filename": "test.mp4",
                "file_size": 1024000,
                "content_type": "video/mp4"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "upload_url" in data
        assert "video_id" in data
        assert data["status"] == "pending_upload"

@pytest.mark.asyncio
async def test_video_view_increment(
    async_client: AsyncClient,
    test_video: Video
):
    """Test that video view count increments."""
    initial_views = test_video.views_count
    
    response = await async_client.post(f"/api/v1/videos/{test_video.id}/view")
    assert response.status_code == 200
    
    # Verify in database
    from app.infrastructure.repositories.video_repository import VideoRepository
    repo = VideoRepository(db_session)
    updated_video = await repo.get(test_video.id)
    assert updated_video.views_count == initial_views + 1
```

---

### 3. API Tests (`tests/api/`)

**Purpose:** Test FastAPI endpoints with full request/response cycle

**Characteristics:**
- Uses TestClient/AsyncClient
- Full middleware stack
- Request validation, response serialization
- Auth flow testing
- Error handling verification

**Examples:**

```python
# tests/api/test_auth_endpoints.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_register_validation_errors(async_client: AsyncClient):
    """Test registration input validation."""
    # Invalid email
    response = await async_client.post("/api/v1/auth/register", json={
        "username": "test",
        "email": "invalid-email",
        "password": "Test123!"
    })
    assert response.status_code == 422
    assert "email" in response.json()["detail"][0]["loc"]
    
    # Weak password
    response = await async_client.post("/api/v1/auth/register", json={
        "username": "test",
        "email": "test@test.com",
        "password": "weak"
    })
    assert response.status_code == 422
    
    # Username too short
    response = await async_client.post("/api/v1/auth/register", json={
        "username": "ab",
        "email": "test@test.com",
        "password": "Test123!"
    })
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_login_incorrect_password(
    async_client: AsyncClient,
    test_user: User
):
    """Test login with incorrect password."""
    response = await async_client.post("/api/v1/auth/login", data={
        "username": test_user.email,
        "password": "WrongPassword"
    })
    assert response.status_code == 401
    assert "incorrect" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_token_refresh(async_client: AsyncClient, refresh_token: str):
    """Test token refresh flow."""
    response = await async_client.post("/api/v1/auth/refresh", json={
        "refresh_token": refresh_token
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data  # New refresh token issued
```

---

### 4. E2E Tests (`tests/e2e/`)

**Purpose:** Test complete user journeys across multiple endpoints

**Characteristics:**
- Multi-step flows
- Real data flow through system
- Critical business scenarios
- Execution time: 100ms - 1s per test
- Coverage target: Critical flows only

**Examples:**

```python
# tests/e2e/test_complete_auth_flow.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_complete_auth_journey(async_client: AsyncClient):
    """
    E2E Test: Register → Login → Access Protected → Refresh Token → Logout
    """
    # Step 1: Register
    register_response = await async_client.post("/api/v1/auth/register", json={
        "username": "e2euser",
        "email": "e2e@test.com",
        "password": "E2ETest123!",
        "full_name": "E2E Test User"
    })
    assert register_response.status_code == 201
    user_id = register_response.json()["id"]
    
    # Step 2: Login
    login_response = await async_client.post("/api/v1/auth/login", data={
        "username": "e2e@test.com",
        "password": "E2ETest123!"
    })
    assert login_response.status_code == 200
    tokens = login_response.json()
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]
    
    # Step 3: Access Protected Endpoint
    me_response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert me_response.status_code == 200
    assert me_response.json()["id"] == user_id
    
    # Step 4: Refresh Token
    refresh_response = await async_client.post("/api/v1/auth/refresh", json={
        "refresh_token": refresh_token
    })
    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()
    assert new_tokens["access_token"] != access_token  # New token issued
    
    # Step 5: Use New Token
    me_response_2 = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {new_tokens['access_token']}"}
    )
    assert me_response_2.status_code == 200


# tests/e2e/test_content_creation_flow.py
@pytest.mark.asyncio
async def test_video_upload_to_view_flow(
    async_client: AsyncClient,
    auth_headers: dict,
    mock_s3,
    mock_transcoding
):
    """
    E2E Test: Upload Video → Process → Publish → View → Stats Update
    """
    # Step 1: Initiate Upload
    init_response = await async_client.post(
        "/api/v1/videos",
        headers=auth_headers,
        json={
            "filename": "e2e_video.mp4",
            "file_size": 5000000,
            "content_type": "video/mp4"
        }
    )
    assert init_response.status_code == 201
    video_id = init_response.json()["video_id"]
    
    # Step 2: Complete Upload (simulate)
    complete_response = await async_client.post(
        f"/api/v1/videos/{video_id}/complete",
        headers=auth_headers,
        json={
            "title": "E2E Test Video",
            "description": "Testing complete flow",
            "tags": ["test", "e2e"],
            "visibility": "public"
        }
    )
    assert complete_response.status_code == 200
    assert complete_response.json()["status"] == "processing"
    
    # Step 3: Simulate Processing Complete (mock webhook or direct update)
    # In real test, would wait for processing or trigger mock completion
    
    # Step 4: Increment View
    view_response = await async_client.post(f"/api/v1/videos/{video_id}/view")
    assert view_response.status_code == 200
    
    # Step 5: Verify Stats
    video_response = await async_client.get(f"/api/v1/videos/{video_id}")
    assert video_response.status_code == 200
    video_data = video_response.json()
    assert video_data["views_count"] > 0
```

---

### 5. Performance Tests (`tests/performance/`)

**Purpose:** Verify system performance under load

**Characteristics:**
- Concurrent request simulation
- Latency measurement (p50, p95, p99)
- Resource utilization
- Load testing

**Examples:**

```python
# tests/performance/test_auth_load.py
import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_logins(async_client: AsyncClient, test_users: list):
    """Test system can handle 100 concurrent login requests."""
    async def login(user):
        response = await async_client.post("/api/v1/auth/login", data={
            "username": user.email,
            "password": "TestPass123!"
        })
        return response.status_code, response.elapsed.total_seconds()
    
    tasks = [login(user) for user in test_users[:100]]
    results = await asyncio.gather(*tasks)
    
    # Verify all succeed
    status_codes = [r[0] for r in results]
    assert status_codes.count(200) >= 95  # 95% success rate
    
    # Verify latency
    latencies = [r[1] for r in results]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    assert p95_latency < 0.5  # p95 < 500ms

@pytest.mark.asyncio
@pytest.mark.slow
async def test_video_list_pagination_performance(async_client: AsyncClient):
    """Test video list endpoint performance with large dataset."""
    import time
    
    start = time.time()
    response = await async_client.get("/api/v1/videos?limit=100")
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 1.0  # < 1 second
    assert len(response.json()["videos"]) <= 100
```

---

### 6. Security Tests (`tests/security/`)

**Purpose:** Verify authorization boundaries and security controls

**Examples:**

```python
# tests/security/test_rbac.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_user_cannot_access_admin_endpoints(
    async_client: AsyncClient,
    user_auth_headers: dict
):
    """Test that regular user cannot access admin endpoints."""
    response = await async_client.get(
        "/api/v1/admin/stats",
        headers=user_auth_headers
    )
    assert response.status_code == 403

@pytest.mark.asyncio
async def test_user_cannot_edit_others_video(
    async_client: AsyncClient,
    user_auth_headers: dict,
    other_user_video: Video
):
    """Test that user cannot edit video owned by another user."""
    response = await async_client.put(
        f"/api/v1/videos/{other_user_video.id}",
        headers=user_auth_headers,
        json={"title": "Hacked"}
    )
    assert response.status_code == 403

@pytest.mark.asyncio
async def test_privilege_escalation_prevention(
    async_client: AsyncClient,
    user_auth_headers: dict,
    db_session: AsyncSession
):
    """Test that user cannot elevate their own role."""
    response = await async_client.put(
        "/api/v1/users/me",
        headers=user_auth_headers,
        json={"role": "admin"}  # Attempt to escalate
    )
    # Should either ignore the role field or return error
    assert response.status_code in [200, 422]
    
    # Verify role didn't change
    me_response = await async_client.get("/api/v1/auth/me", headers=user_auth_headers)
    assert me_response.json()["role"] != "admin"

@pytest.mark.asyncio
async def test_xss_prevention_in_posts(
    async_client: AsyncClient,
    auth_headers: dict
):
    """Test that XSS payload is sanitized."""
    xss_payload = "<script>alert('XSS')</script>"
    
    response = await async_client.post(
        "/api/v1/social/posts",
        headers=auth_headers,
        json={"content": xss_payload}
    )
    assert response.status_code == 201
    
    post_id = response.json()["id"]
    get_response = await async_client.get(f"/api/v1/social/posts/{post_id}")
    
    content = get_response.json()["content"]
    assert "<script>" not in content  # Script tags removed/escaped
```

---

## Test Fixtures (`conftest.py`)

```python
# tests/conftest.py
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import get_db, Base
from app.core.config import settings
from app.models.user import User, UserRole, UserStatus
from app.core.security import get_password_hash

# Test database setup
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

@pytest_asyncio.fixture
async def db_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(db_engine):
    """Create test database session with automatic rollback."""
    async_session = sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        async with session.begin():
            yield session
            await session.rollback()

@pytest_asyncio.fixture
async def async_client(db_session):
    """Create async HTTP client for testing."""
    # Override get_db dependency
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def test_user(db_session):
    """Create test user."""
    user = User(
        username="testuser",
        email="test@test.com",
        password_hash=get_password_hash("TestPass123!"),
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest_asyncio.fixture
async def admin_user(db_session):
    """Create admin user."""
    user = User(
        username="admin",
        email="admin@test.com",
        password_hash=get_password_hash("AdminPass123!"),
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user

@pytest_asyncio.fixture
async def auth_headers(async_client, test_user):
    """Create authentication headers."""
    response = await async_client.post("/api/v1/auth/login", data={
        "username": test_user.email,
        "password": "TestPass123!"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest_asyncio.fixture
async def admin_auth_headers(async_client, admin_user):
    """Create admin authentication headers."""
    response = await async_client.post("/api/v1/auth/login", data={
        "username": admin_user.email,
        "password": "AdminPass123!"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def mock_s3(monkeypatch):
    """Mock S3 operations."""
    from unittest.mock import AsyncMock
    
    async def mock_upload(*args, **kwargs):
        return {"key": "test-key", "url": "https://s3.amazonaws.com/test"}
    
    monkeypatch.setattr("app.services.storage_service.StorageService.upload", mock_upload)
    return mock_upload

@pytest.fixture
def mock_stripe(monkeypatch):
    """Mock Stripe operations."""
    from unittest.mock import Mock
    
    mock_customer = Mock()
    mock_customer.id = "cus_test123"
    
    monkeypatch.setattr("stripe.Customer.create", lambda **kwargs: mock_customer)
    return mock_customer
```

---

## Coverage Goals

| Category | Target | Rationale |
|----------|--------|-----------|
| Overall | 80% | Industry standard |
| Critical Paths | 95% | Business-critical |
| Models | 90% | Core data layer |
| Services | 85% | Business logic |
| API Endpoints | 90% | User-facing |
| Utils | 80% | Helper functions |

---

## Test Execution

```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest tests/ --cov=app --cov-report=html --cov-report=term

# Run fast tests only (exclude slow performance tests)
pytest tests/ -m "not slow"

# Run security tests
pytest tests/security/

# Run specific test file
pytest tests/unit/test_user_model.py

# Run with verbose output
pytest tests/ -v

# Run with parallel execution (8 workers)
pytest tests/ -n 8

# Run and stop on first failure
pytest tests/ -x
```

---

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest tests/ --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

**Next: See observability_plan.md for logging and metrics strategy**
