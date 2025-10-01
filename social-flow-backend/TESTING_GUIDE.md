# Testing Guide - Social Flow Backend

## Overview

This document provides comprehensive information about the testing infrastructure for the Social Flow backend. We have implemented a multi-layered testing strategy targeting **85% code coverage** with unit tests, integration tests, performance tests, and security tests.

## Table of Contents

1. [Test Structure](#test-structure)
2. [Running Tests](#running-tests)
3. [Test Types](#test-types)
4. [Coverage Reports](#coverage-reports)
5. [Writing Tests](#writing-tests)
6. [Continuous Integration](#continuous-integration)
7. [Best Practices](#best-practices)

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── run_tests.py                # Test runner script
├── unit/                       # Unit tests (individual components)
│   ├── test_auth.py           # Authentication service tests
│   ├── test_post_service.py   # Post service tests
│   ├── test_payment_service.py # Payment service tests
│   ├── test_ml_service.py     # ML service tests
│   ├── test_video.py          # Video service tests
│   └── test_ml.py             # ML component tests
├── integration/                # Integration tests (API endpoints)
│   ├── test_auth_integration.py
│   ├── test_post_api.py
│   ├── test_payment_api.py
│   └── test_video_integration.py
├── performance/                # Performance/load tests
│   └── locustfile.py          # Locust load testing scenarios
└── security/                   # Security tests
    └── test_security.py       # Security vulnerability tests
```

## Running Tests

### Prerequisites

Install test dependencies:

```powershell
pip install -r requirements-dev.txt
```

### Quick Start

```powershell
# Run all tests
python tests/run_tests.py all

# Run specific test types
python tests/run_tests.py unit           # Unit tests only
python tests/run_tests.py integration    # Integration tests only
python tests/run_tests.py security       # Security tests only
python tests/run_tests.py performance    # Performance tests (Locust)

# Run with verbose output
python tests/run_tests.py all -v

# Run with coverage report
python tests/run_tests.py unit -c
```

### Using pytest directly

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_post_service.py

# Run specific test class
pytest tests/unit/test_post_service.py::TestPostService

# Run specific test method
pytest tests/unit/test_post_service.py::TestPostService::test_create_post_success

# Run tests by marker
pytest -m unit              # Unit tests
pytest -m integration       # Integration tests
pytest -m security          # Security tests
pytest -m fast              # Fast tests
pytest -m slow              # Slow tests
pytest -m auth              # Authentication tests
pytest -m payment           # Payment tests

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Run in parallel (faster)
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l
```

## Test Types

### 1. Unit Tests

**Location**: `tests/unit/`

**Purpose**: Test individual components in isolation using mocks and stubs.

**Coverage Target**: 85%

**Examples**:
- Service layer logic
- Utility functions
- Data transformations
- Business logic

**Sample Test**:
```python
@pytest.mark.asyncio
async def test_create_post_success(post_service, mock_db, test_user):
    """Test successful post creation."""
    post_data = PostCreate(
        content="Test post #hashtag",
        visibility="public",
    )
    
    result = await post_service.create_post(post_data, test_user.id)
    
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    assert result.content == post_data.content
```

### 2. Integration Tests

**Location**: `tests/integration/`

**Purpose**: Test API endpoints with real database interactions.

**Examples**:
- API endpoint functionality
- Request/response validation
- Database operations
- Authentication flows

**Sample Test**:
```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_create_post_success(async_client: AsyncClient, test_user: User, auth_headers: dict):
    """Test creating a post via API."""
    post_data = {
        "content": "Test post",
        "visibility": "public",
    }
    
    response = await async_client.post(
        "/api/v1/posts/",
        json=post_data,
        headers=auth_headers,
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
```

### 3. Performance Tests

**Location**: `tests/performance/`

**Purpose**: Load testing and performance benchmarking using Locust.

**Running Performance Tests**:

```powershell
# Start Locust web UI
python tests/run_tests.py performance

# Or directly with locust
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser to configure and run tests.

**Test Scenarios**:
- Video streaming load (target: 1000 concurrent users)
- Feed generation (target: 500 req/s)
- Authentication (target: 200 req/s)
- Live streaming viewers
- Search operations

**Performance Targets**:
- **API Response Time**: < 200ms (p95)
- **Video Streaming**: 1000+ concurrent streams
- **Feed Generation**: 500+ requests/second
- **Database Queries**: < 50ms (p95)
- **Error Rate**: < 0.1%

### 4. Security Tests

**Location**: `tests/security/`

**Purpose**: Verify protection against common vulnerabilities.

**Test Coverage**:
- SQL injection prevention
- XSS (Cross-Site Scripting) prevention
- Authentication bypass attempts
- Authorization checks
- Rate limiting
- CSRF protection
- Input validation
- Path traversal prevention
- Session security

**Sample Test**:
```python
@pytest.mark.asyncio
@pytest.mark.security
async def test_sql_injection_in_login(async_client: AsyncClient):
    """Test SQL injection prevention."""
    response = await async_client.post(
        "/api/v1/auth/login",
        json={
            "email": "admin' OR '1'='1",
            "password": "password",
        }
    )
    assert response.status_code in [400, 401, 422]
```

## Coverage Reports

### Generate Coverage Report

```powershell
# Generate HTML and terminal report
pytest --cov=app --cov-report=html --cov-report=term-missing

# Open HTML report
start htmlcov/index.html  # Windows
```

### Coverage Goals

- **Overall**: 85% code coverage
- **Services**: 90% coverage
- **API Endpoints**: 85% coverage
- **Models**: 80% coverage
- **Utilities**: 90% coverage

### Viewing Coverage

The HTML coverage report shows:
- Line-by-line coverage
- Uncovered lines highlighted
- Branch coverage
- Function coverage

## Writing Tests

### Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
# Database fixtures
@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Create test database session."""
    # Creates isolated test database

# Client fixtures
@pytest_asyncio.fixture
async def async_client(db_session: AsyncSession) -> AsyncClient:
    """Create async test client."""
    # Returns async HTTP client

# Model fixtures
@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create test user."""
    # Returns test user instance

@pytest_asyncio.fixture
async def test_post(db_session: AsyncSession, test_user: User) -> Post:
    """Create test post."""
    # Returns test post instance

# Authentication fixtures
@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Create authentication headers."""
    return {"Authorization": f"Bearer test_token_{test_user.id}"}
```

### Test Naming Convention

- **Test files**: `test_<module_name>.py`
- **Test classes**: `Test<ComponentName>`
- **Test methods**: `test_<action>_<expected_result>`

Examples:
- `test_create_post_success`
- `test_update_post_unauthorized`
- `test_delete_post_not_found`

### Test Structure (AAA Pattern)

```python
@pytest.mark.asyncio
async def test_example():
    # Arrange - Set up test data
    user = create_test_user()
    post_data = {"content": "Test"}
    
    # Act - Perform the action
    result = await post_service.create_post(post_data, user.id)
    
    # Assert - Verify the outcome
    assert result.content == "Test"
    assert result.owner_id == user.id
```

### Mocking External Services

```python
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_with_mock():
    with patch('stripe.PaymentIntent.create') as mock_stripe:
        mock_stripe.return_value = {"id": "pi_test123"}
        
        result = await payment_service.create_payment_intent(1000, "USD")
        
        assert result["id"] == "pi_test123"
        mock_stripe.assert_called_once()
```

### Async Testing

Use `@pytest.mark.asyncio` for async tests:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

## Continuous Integration

### GitHub Actions Workflow

The CI pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests

**Pipeline Steps**:
1. Install dependencies
2. Run linters (black, isort, flake8, mypy)
3. Run unit tests with coverage
4. Run integration tests
5. Run security tests
6. Generate coverage report
7. Fail if coverage < 85%

### Pre-commit Hooks

Install pre-commit hooks:

```powershell
pre-commit install
```

Hooks will run automatically on `git commit`:
- Code formatting (black, isort)
- Linting (flake8)
- Security checks (bandit)

## Best Practices

### 1. Test Independence

Each test should be independent and not rely on other tests:

```python
# ✅ Good - Independent test
@pytest.mark.asyncio
async def test_create_post(db_session):
    user = await create_test_user(db_session)
    post = await create_post(user.id)
    assert post.owner_id == user.id

# ❌ Bad - Depends on global state
user = None

async def test_setup():
    global user
    user = await create_test_user()

async def test_create_post():
    post = await create_post(user.id)  # Depends on previous test
```

### 2. Use Fixtures for Setup

```python
# ✅ Good - Use fixtures
@pytest.mark.asyncio
async def test_with_fixture(test_user, test_post):
    result = await post_service.get_post(test_post.id)
    assert result.owner_id == test_user.id

# ❌ Bad - Manual setup in each test
@pytest.mark.asyncio
async def test_without_fixture(db_session):
    user = User(username="test", email="test@example.com")
    db_session.add(user)
    await db_session.commit()
    # ... more setup code repeated in every test
```

### 3. Test Edge Cases

```python
# Test normal case
async def test_divide_normal():
    assert divide(10, 2) == 5

# Test edge cases
async def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

async def test_divide_negative():
    assert divide(-10, 2) == -5
```

### 4. Clear Test Names

```python
# ✅ Good - Descriptive name
async def test_create_post_with_empty_content_returns_validation_error():
    pass

# ❌ Bad - Vague name
async def test_post():
    pass
```

### 5. Use Markers

```python
@pytest.mark.unit
@pytest.mark.fast
async def test_quick_validation():
    pass

@pytest.mark.integration
@pytest.mark.slow
async def test_full_workflow():
    pass

@pytest.mark.skip(reason="API not yet implemented")
async def test_future_feature():
    pass
```

### 6. Test Error Cases

```python
# Test success case
async def test_login_success():
    result = await auth_service.login("user@example.com", "password")
    assert result["access_token"]

# Test error cases
async def test_login_invalid_email():
    with pytest.raises(ValidationError):
        await auth_service.login("invalid-email", "password")

async def test_login_wrong_password():
    with pytest.raises(AuthenticationError):
        await auth_service.login("user@example.com", "wrong")
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```powershell
# Ensure you're in the project root
cd c:\Users\nirma\OneDrive\Desktop\social-flow-main\social-flow-backend

# Set PYTHONPATH
$env:PYTHONPATH = "."
```

**2. Database Connection Errors**

Tests use SQLite in-memory database by default. Ensure `conftest.py` is configured correctly.

**3. Async Test Failures**

Make sure to use `@pytest.mark.asyncio` decorator:

```python
@pytest.mark.asyncio  # Required!
async def test_async_function():
    result = await some_async_function()
    assert result
```

**4. Mock Not Working**

Ensure you're patching the correct import path:

```python
# ✅ Good - Patch where it's used
with patch('app.services.payment.stripe.PaymentIntent.create'):
    pass

# ❌ Bad - Patch where it's defined
with patch('stripe.PaymentIntent.create'):
    pass
```

## Test Metrics

### Current Coverage

Run to see current coverage:

```powershell
pytest --cov=app --cov-report=term-missing
```

### Target Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Overall Coverage | 85% | TBD |
| Unit Test Coverage | 90% | TBD |
| Integration Test Coverage | 85% | TBD |
| Test Execution Time | < 5 min | TBD |
| Performance Test Pass Rate | 100% | TBD |
| Security Test Pass Rate | 100% | TBD |

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Locust Documentation](https://docs.locust.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

## Support

For questions or issues with testing:
1. Check this documentation
2. Review existing test examples in `tests/`
3. Check pytest documentation
4. Create an issue in the repository
