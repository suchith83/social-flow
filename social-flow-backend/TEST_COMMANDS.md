# 🧪 Social Flow Backend - Complete Testing Guide

> **All commands to test the Social Flow Backend application**

---

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [API Endpoint Tests](#api-endpoint-tests)
- [Coverage Reports](#coverage-reports)
- [Specific Module Tests](#specific-module-tests)
- [Performance Tests](#performance-tests)
- [Security Tests](#security-tests)
- [Continuous Integration](#continuous-integration)

---

## Environment Setup

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Set Up Test Database

```bash
# Run database migrations
alembic upgrade head

# Or use PowerShell script
.\setup-database.ps1
```

---

## Unit Tests

### Run All Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run with verbose output
pytest tests/unit/ -v

# Run with extra verbose output (show all test names)
pytest tests/unit/ -vv

# Run with short test summary
pytest tests/unit/ -v --tb=short
```

### Run Tests with Output

```bash
# Show print statements
pytest tests/unit/ -s

# Show print statements with verbose
pytest tests/unit/ -sv
```

### Run Tests in Parallel

```bash
# Run tests in parallel (faster)
pytest tests/unit/ -n auto

# Run tests on 4 cores
pytest tests/unit/ -n 4
```

### Stop on First Failure

```bash
# Stop at first failure
pytest tests/unit/ -x

# Stop after 3 failures
pytest tests/unit/ --maxfail=3
```

---

## Integration Tests

### Run Integration Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run specific integration test file
pytest tests/integration/test_api_integration.py

# Run with fixtures
pytest tests/integration/ -v --setup-show
```

### Run Advanced Integration Tests

```bash
# Run advanced integration tests
pytest advanced_integration_test.py

# Run with JSON output
pytest advanced_integration_test.py --json-report --json-report-file=test_results.json
```

### Run Infrastructure Tests

```bash
# Test infrastructure components
pytest test_infrastructure.py -v

# Test with detailed output
python test_infrastructure.py
```

---

## API Endpoint Tests

### Run API Tests

```bash
# Run API endpoint tests
pytest test_api_endpoints.py

# Run with verbose output
pytest test_api_endpoints.py -v

# Run specific test
pytest test_api_endpoints.py::test_health_endpoint

# Run and save output
pytest test_api_endpoints.py -v > test_output.txt 2>&1
```

### Manual API Testing

```bash
# Using Python script
python test_api_endpoints.py

# Using curl for quick checks
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

---

## Coverage Reports

### Run Tests with Coverage

```bash
# Run all tests with coverage
pytest tests/unit/ --cov=app --cov-report=term

# Generate HTML coverage report
pytest tests/unit/ --cov=app --cov-report=html --cov-report=term

# Generate multiple report formats
pytest tests/unit/ --cov=app --cov-report=html --cov-report=term --cov-report=xml
```

### View Coverage Report

```bash
# Open HTML coverage report (Windows)
start htmlcov\index.html

# Open HTML coverage report (Linux/Mac)
open htmlcov/index.html

# Or navigate to: file:///path/to/social-flow-backend/htmlcov/index.html
```

### Coverage with Missing Lines

```bash
# Show missing lines in terminal
pytest tests/unit/ --cov=app --cov-report=term-missing

# Show missing lines with verbose
pytest tests/unit/ --cov=app --cov-report=term-missing:skip-covered
```

### Specific Module Coverage

```bash
# Coverage for specific module
pytest tests/unit/ --cov=app.auth --cov-report=term

# Coverage for multiple modules
pytest tests/unit/ --cov=app.auth --cov=app.users --cov-report=html
```

---

## Specific Module Tests

### Authentication Tests

```bash
# Run all auth tests
pytest tests/unit/auth/ -v

# Run specific auth test file
pytest tests/unit/auth/test_auth_service.py

# Run specific test function
pytest tests/unit/auth/test_auth_service.py::TestAuthService::test_register_user_with_verification_success

# Run with coverage
pytest tests/unit/auth/ --cov=app.auth --cov-report=term
```

### User Tests

```bash
# Run all user tests
pytest tests/unit/users/ -v

# Test user service
pytest tests/unit/test_user_service.py

# Test with coverage
pytest tests/unit/users/ --cov=app.users --cov-report=html
```

### Video Tests

```bash
# Run all video tests
pytest tests/unit/videos/ -v

# Test video upload
pytest tests/unit/videos/test_video_upload.py

# Test video streaming
pytest tests/unit/videos/test_video_streaming.py
```

### Social Tests

```bash
# Run all social tests
pytest tests/unit/social/ -v

# Test posts
pytest tests/unit/test_posts.py

# Test comments
pytest tests/unit/test_comments.py

# Test likes
pytest tests/unit/test_likes.py
```

### Payment Tests

```bash
# Run all payment tests
pytest tests/unit/payments/ -v

# Test Stripe integration
pytest tests/unit/test_stripe_integration.py

# Test subscriptions
pytest tests/unit/test_subscriptions.py
```

### ML/AI Tests

```bash
# Run all ML tests
pytest tests/unit/test_ml.py -v

# Test recommendation engine
pytest tests/unit/test_recommendations.py

# Test content moderation
pytest tests/unit/test_moderation.py

# Test with longer timeout for ML operations
pytest tests/unit/test_ml.py --timeout=300
```

### Notification Tests

```bash
# Run notification tests
pytest tests/unit/test_notifications.py -v

# Test push notifications
pytest tests/unit/test_push_notifications.py
```

---

## Performance Tests

### Load Testing

```bash
# Run performance tests
pytest tests/performance/ -v

# Run with timing
pytest tests/unit/ --durations=10

# Show slowest tests
pytest tests/unit/ --durations=0
```

### Benchmark Tests

```bash
# Install pytest-benchmark
pip install pytest-benchmark

# Run benchmark tests
pytest tests/benchmark/ --benchmark-only

# Compare benchmarks
pytest tests/benchmark/ --benchmark-compare
```

---

## Security Tests

### Run Security Scans

```bash
# Run Bandit security scanner
bandit -r app/ -f json -o bandit_report.json

# Run with verbose output
bandit -r app/ -v

# Scan specific severity levels
bandit -r app/ -ll  # Low level and above
bandit -r app/ -lll # High level only
```

### Safety Check

```bash
# Check for known security vulnerabilities
safety check

# Check with detailed output
safety check --json

# Check specific file
safety check -r requirements.txt
```

---

## Test Markers

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run slow tests
pytest -m slow

# Run fast tests only
pytest -m "not slow"

# Run database tests
pytest -m database

# Run API tests
pytest -m api
```

### Available Markers

```bash
# List all markers
pytest --markers

# Common markers in this project:
# - unit: Unit tests
# - integration: Integration tests
# - slow: Slow-running tests
# - fast: Fast tests
# - database: Database tests
# - api: API endpoint tests
# - auth: Authentication tests
# - payment: Payment tests
# - ml: Machine learning tests
```

---

## Test Selection

### Run by Test Name Pattern

```bash
# Run tests matching pattern
pytest -k "test_auth"

# Run tests with specific word
pytest -k "login"

# Run tests NOT matching pattern
pytest -k "not slow"

# Multiple patterns (OR)
pytest -k "auth or user"

# Multiple patterns (AND)
pytest -k "auth and login"
```

### Run by File Pattern

```bash
# Run all test files matching pattern
pytest tests/unit/test_auth*.py

# Run tests in specific directory
pytest tests/unit/auth/
```

---

## Test Reporting

### Generate Test Reports

```bash
# Generate JUnit XML report
pytest tests/unit/ --junitxml=test-results.xml

# Generate JSON report (requires pytest-json-report)
pip install pytest-json-report
pytest tests/unit/ --json-report --json-report-file=report.json

# Generate HTML report (requires pytest-html)
pip install pytest-html
pytest tests/unit/ --html=report.html --self-contained-html
```

### Test Output Formats

```bash
# Detailed output
pytest tests/unit/ -v

# Very detailed output
pytest tests/unit/ -vv

# Quiet output (only show failures)
pytest tests/unit/ -q

# No output capture (show print statements)
pytest tests/unit/ -s
```

---

## Continuous Integration

### GitHub Actions / CI Pipeline

```bash
# Run full CI test suite
pytest tests/unit/ --cov=app --cov-report=xml --cov-report=html --junitxml=test-results.xml

# Run with all checks
pytest tests/unit/ -v --cov=app --cov-report=term-missing --cov-fail-under=39
```

### Pre-commit Checks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run mypy --all-files
```

---

## Code Quality Checks

### Type Checking

```bash
# Run mypy type checker
mypy app/

# Run with specific config
mypy app/ --config-file=mypy.ini

# Check specific module
mypy app/auth/
```

### Code Formatting

```bash
# Format code with Black
black app/

# Check formatting without changes
black app/ --check

# Format with line length
black app/ --line-length=100
```

### Linting

```bash
# Run flake8 linter
flake8 app/

# Run with specific config
flake8 app/ --config=.flake8

# Run with statistics
flake8 app/ --statistics
```

### Import Sorting

```bash
# Sort imports with isort
isort app/

# Check import order
isort app/ --check-only

# Show diff
isort app/ --diff
```

---

## Database Tests

### Test Database Operations

```bash
# Run database-related tests
pytest tests/unit/ -m database

# Test migrations
pytest tests/unit/test_migrations.py

# Test with real database
pytest tests/integration/ --db=postgresql
```

### Reset Test Database

```bash
# Drop and recreate test database
alembic downgrade base
alembic upgrade head

# Or use script
python scripts/reset_test_db.py
```

---

## Docker Tests

### Test in Docker Container

```bash
# Build test image
docker build -t social-flow-test -f Dockerfile .

# Run tests in container
docker run social-flow-test pytest tests/unit/

# Run with volume mount
docker run -v $(pwd):/app social-flow-test pytest tests/unit/
```

### Docker Compose Tests

```bash
# Run tests with docker-compose
docker-compose -f docker-compose.yml run --rm app pytest tests/unit/

# Run with coverage
docker-compose run --rm app pytest tests/unit/ --cov=app --cov-report=html
```

---

## Debugging Tests

### Run Tests in Debug Mode

```bash
# Run with pdb debugger on failure
pytest tests/unit/ --pdb

# Drop into debugger on first failure
pytest tests/unit/ -x --pdb

# Run specific test with breakpoint
pytest tests/unit/test_auth.py::test_login --pdb
```

### Verbose Debugging

```bash
# Show local variables on failure
pytest tests/unit/ -l

# Show full diff
pytest tests/unit/ --tb=long

# Show captured output on failure
pytest tests/unit/ --tb=short -s
```

---

## Test Configuration

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=app
    --cov-report=term-missing
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow-running tests
    api: API endpoint tests
```

### conftest.py Fixtures

```bash
# View available fixtures
pytest --fixtures

# View specific fixture
pytest --fixtures -v

# Use fixture in test
pytest tests/unit/test_auth.py --setup-show
```

---

## Quick Test Recipes

### 1. Quick Smoke Test
```bash
# Run fast tests only
pytest tests/unit/ -m "not slow" -q
```

### 2. Full Test Suite
```bash
# Run everything with coverage
pytest tests/unit/ --cov=app --cov-report=html -v
```

### 3. Failed Tests Only
```bash
# Run only previously failed tests
pytest tests/unit/ --lf

# Run failed first, then others
pytest tests/unit/ --ff
```

### 4. Changed Code Only
```bash
# Test only changed files (requires pytest-picked)
pip install pytest-picked
pytest --picked
```

### 5. Watch Mode
```bash
# Auto-run tests on file changes (requires pytest-watch)
pip install pytest-watch
ptw tests/unit/ -- -v
```

---

## Environment Variables for Testing

```bash
# Set test environment
export ENVIRONMENT=test
export DATABASE_URL=sqlite+aiosqlite:///./test.db
export TESTING=true

# Windows PowerShell
$env:ENVIRONMENT="test"
$env:DATABASE_URL="sqlite+aiosqlite:///./test.db"
$env:TESTING="true"
```

---

## Test Statistics

### Get Test Statistics

```bash
# Count tests
pytest tests/unit/ --collect-only -q

# Show test durations
pytest tests/unit/ --durations=0

# Show test coverage summary
pytest tests/unit/ --cov=app --cov-report=term | tail -n 20
```

---

## Common Test Commands Summary

```bash
# 1. Run all tests
pytest tests/unit/

# 2. Run with coverage
pytest tests/unit/ --cov=app --cov-report=html

# 3. Run specific module
pytest tests/unit/auth/

# 4. Run with verbose output
pytest tests/unit/ -v

# 5. Run fast tests only
pytest tests/unit/ -m "not slow"

# 6. Run and stop on first failure
pytest tests/unit/ -x

# 7. Run in parallel
pytest tests/unit/ -n auto

# 8. Run with debugging
pytest tests/unit/ --pdb

# 9. Generate HTML report
pytest tests/unit/ --html=report.html

# 10. Full CI pipeline
pytest tests/unit/ --cov=app --cov-report=xml --junitxml=results.xml -v
```

---

## Test Results Interpretation

### Reading Coverage Output

```
---------- coverage: platform windows, python 3.11.x -----------
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
app/__init__.py                      10      0   100%
app/auth/service.py                 150     30    80%   45-50, 67-70
app/users/service.py                120     60    50%   23-45, 89-120
---------------------------------------------------------------
TOTAL                              7568   4042    39%
```

- **Stmts**: Total statements in file
- **Miss**: Statements not covered by tests
- **Cover**: Percentage covered
- **Missing**: Line numbers not covered

### Test Result Symbols

- `.` - Test passed
- `F` - Test failed
- `E` - Test error
- `s` - Test skipped
- `x` - Expected failure
- `X` - Unexpected pass

---

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pytest with src layout
pytest tests/unit/ --import-mode=importlib
```

**2. Database Connection Errors**
```bash
# Use test database
pytest tests/unit/ --db-url=sqlite:///test.db

# Reset database before tests
alembic downgrade base && alembic upgrade head
```

**3. Fixture Not Found**
```bash
# Check fixture scope
pytest --fixtures | grep fixture_name

# Ensure conftest.py is in correct location
ls tests/conftest.py
```

**4. Slow Tests**
```bash
# Identify slow tests
pytest tests/unit/ --durations=10

# Skip slow tests
pytest tests/unit/ -m "not slow"
```

---

## Test Best Practices

1. **Run tests frequently** during development
2. **Use markers** to organize and filter tests
3. **Maintain test data** separate from production data
4. **Mock external services** (Stripe, AWS, etc.)
5. **Test edge cases** and error conditions
6. **Keep tests fast** - use fixtures and factories
7. **Use meaningful test names** - describe what is tested
8. **Test one thing** per test function
9. **Use parametrize** for multiple similar test cases
10. **Clean up after tests** - use fixtures with cleanup

---

## Additional Resources

- **pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Project Test Report**: See `TEST_ACHIEVEMENT_REPORT.md`
- **Coverage Roadmap**: See `COVERAGE_ROADMAP.md`
- **Testing Guide**: See `PHASE_7_8_TESTING_GUIDE.md`

---

**Last Updated:** October 5, 2025  
**Test Pass Rate:** 500/500 (100%)  
**Coverage:** 39% (Target: 60-70%)  
**Maintained by:** Social Flow Development Team
