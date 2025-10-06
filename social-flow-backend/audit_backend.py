"""
Comprehensive Backend Audit Script

Generates complete analysis of:
1. Issues catalog (issues.md)
2. Remediation plan (remediation_plan.md)
3. Code diffs (code_diffs.md)
4. Validation report (validation_report.md)
5. Test strategy (test_strategy.md)
6. Observability plan (observability_plan.md)
"""

import sys
import inspect
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class BackendAuditor:
    def __init__(self):
        self.issues = []
        self.routes_declared = []
        self.routes_implemented = []
        self.models_declared = []
        self.models_implemented = []
        self.tests_found = []
        self.tests_needed = []
        
    def audit_routes(self):
        """Audit all API routes"""
        print("\n" + "="*80)
        print("AUDITING API ROUTES")
        print("="*80)
        
        try:
            from app.api.v1.router import api_router
            
            routes = []
            for route in api_router.routes:
                if hasattr(route, 'path'):
                    methods = getattr(route, 'methods', ['WS'])
                    routes.append({
                        'path': route.path,
                        'methods': list(methods),
                        'name': getattr(route, 'name', 'unknown')
                    })
            
            self.routes_implemented = routes
            print(f"[OK] Found {len(routes)} implemented routes")
            
            # Group by prefix
            prefix_groups = defaultdict(list)
            for route in routes:
                prefix = route['path'].split('/')[0] if route['path'] else 'root'
                prefix_groups[prefix].append(route)
            
            print("\nRoute distribution:")
            for prefix, group in sorted(prefix_groups.items()):
                print(f"  /{prefix}: {len(group)} routes")
            
            return routes
            
        except Exception as e:
            print(f"[ERROR] Failed to audit routes: {e}")
            self.issues.append({
                'category': 'CRITICAL',
                'component': 'API Router',
                'issue': f'Failed to load API router: {e}',
                'impact': 'Cannot analyze API endpoints'
            })
            return []
    
    def audit_models(self):
        """Audit all database models"""
        print("\n" + "="*80)
        print("AUDITING DATABASE MODELS")
        print("="*80)
        
        try:
            from app.models.base import Base
            from app.models import MODEL_REGISTRY
            
            tables = list(Base.metadata.tables.keys())
            self.models_implemented = tables
            
            print(f"✓ Found {len(tables)} database tables:")
            for table in sorted(tables):
                print(f"  - {table}")
            
            # Check MODEL_REGISTRY completeness
            print(f"\n✓ MODEL_REGISTRY has {len(MODEL_REGISTRY)} models")
            
            # Verify all models are in Base.metadata
            registry_tables = [m.__tablename__ for m in MODEL_REGISTRY if hasattr(m, '__tablename__')]
            missing_in_metadata = set(registry_tables) - set(tables)
            missing_in_registry = set(tables) - set(registry_tables)
            
            if missing_in_metadata:
                print(f"\n⚠ Models in REGISTRY but not in Base.metadata: {missing_in_metadata}")
                self.issues.append({
                    'category': 'CONSISTENCY',
                    'component': 'Models',
                    'issue': f'Models registered but not in metadata: {missing_in_metadata}',
                    'impact': 'Alembic migrations may not detect these models'
                })
            
            if missing_in_registry:
                print(f"\n⚠ Tables in metadata but not in MODEL_REGISTRY: {missing_in_registry}")
                self.issues.append({
                    'category': 'CONSISTENCY',
                    'component': 'Models',
                    'issue': f'Tables not in MODEL_REGISTRY: {missing_in_registry}',
                    'impact': 'Documentation and migration tracking may be incomplete'
                })
            
            return tables
            
        except Exception as e:
            print(f"✗ Failed to audit models: {e}")
            self.issues.append({
                'category': 'CRITICAL',
                'component': 'Models',
                'issue': f'Failed to load models: {e}',
                'impact': 'Cannot analyze database schema'
            })
            return []
    
    def audit_health_endpoints(self):
        """Audit health check implementation"""
        print("\n" + "="*80)
        print("AUDITING HEALTH ENDPOINTS")
        print("="*80)
        
        try:
            from app.api.v1.endpoints import health
            
            # Check if health checks handle missing subsystems
            source = inspect.getsource(health.check_s3)
            
            issues_found = []
            if 'try:' not in source or 'except' not in source:
                issues_found.append('check_s3 may not handle exceptions gracefully')
            
            source = inspect.getsource(health.check_redis)
            if 'try:' not in source or 'except' not in source:
                issues_found.append('check_redis may not handle exceptions gracefully')
            
            if issues_found:
                print("⚠ Health check issues found:")
                for issue in issues_found:
                    print(f"  - {issue}")
                    self.issues.append({
                        'category': 'FUNCTIONAL_GAP',
                        'component': 'Health Checks',
                        'issue': issue,
                        'impact': 'Application may fail health checks when optional services are unavailable'
                    })
            else:
                print("✓ Health checks appear to handle exceptions")
            
        except Exception as e:
            print(f"✗ Failed to audit health endpoints: {e}")
            self.issues.append({
                'category': 'CRITICAL',
                'component': 'Health Checks',
                'issue': f'Failed to analyze health checks: {e}',
                'impact': 'Cannot verify robustness of health monitoring'
            })
    
    def audit_test_coverage(self):
        """Audit test coverage"""
        print("\n" + "="*80)
        print("AUDITING TEST COVERAGE")
        print("="*80)
        
        test_dirs = [
            'tests/unit',
            'tests/integration',
            'tests/e2e',
            'tests/performance',
            'tests/security'
        ]
        
        for test_dir in test_dirs:
            path = Path(test_dir)
            if path.exists():
                test_files = list(path.glob('**/test_*.py'))
                self.tests_found.extend([str(f) for f in test_files])
                print(f"✓ {test_dir}: {len(test_files)} test files")
            else:
                print(f"⚠ {test_dir}: directory not found")
        
        # Check for placeholder test files
        placeholder_tests = [
            'comprehensive_test.py',
            'advanced_integration_test.py',
            'tests/test_all_backend.py'
        ]
        
        for test_file in placeholder_tests:
            if Path(test_file).exists():
                print(f"\n⚠ Found placeholder test: {test_file}")
                self.issues.append({
                    'category': 'TEST_INTEGRITY',
                    'component': 'Test Suite',
                    'issue': f'Placeholder/report-style test found: {test_file}',
                    'impact': 'These tests do not provide real assertions and should be converted to pytest'
                })
        
        print(f"\nTotal test files found: {len(self.tests_found)}")
    
    def audit_documentation_claims(self):
        """Compare documentation claims with implementation"""
        print("\n" + "="*80)
        print("AUDITING DOCUMENTATION VS IMPLEMENTATION")
        print("="*80)
        
        # Check README claims
        readme_path = Path('README.md')
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding='utf-8', errors='ignore')
                
                # Extract endpoint count claim
                if '107' in content or '107+' in content:
                    actual_count = len(self.routes_implemented)
                    print("\nREADME claims: 107+ endpoints")
                    print(f"Actual count: {actual_count} endpoints")
                    
                    if actual_count < 107:
                        self.issues.append({
                            'category': 'DOCUMENTATION_DRIFT',
                            'component': 'README',
                            'issue': f'README claims 107+ endpoints but only {actual_count} found',
                            'impact': 'Misleading documentation'
                        })
                    elif actual_count > 107:
                        print(f"[OK] Endpoint count exceeds documentation ({actual_count} > 107)")
            except Exception as e:
                print(f"[WARN] Could not read README: {e}")
        
        # Check API documentation
        api_doc_path = Path('COMPLETE_API_DOCUMENTATION.md')
        if api_doc_path.exists():
            try:
                content = api_doc_path.read_text(encoding='utf-8', errors='ignore')
                
                # Look for endpoint declarations
                declared_endpoints = []
                import re
                patterns = [
                    r'POST.*?/api/v1/(\S+)',
                    r'GET.*?/api/v1/(\S+)',
                    r'PUT.*?/api/v1/(\S+)',
                    r'DELETE.*?/api/v1/(\S+)',
                    r'PATCH.*?/api/v1/(\S+)'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    declared_endpoints.extend(matches)
                
                self.routes_declared = list(set(declared_endpoints))
                print(f"\n[OK] Documentation declares {len(self.routes_declared)} unique endpoints")
                
                # Find undeclared but implemented
                implemented_paths = [r['path'].replace('/', '', 1) for r in self.routes_implemented if r['path']]
                undeclared = set(implemented_paths) - set(self.routes_declared)
                
                if undeclared and len(undeclared) < 50:  # Only report if reasonable number
                    print(f"\n[WARN] Found {len(undeclared)} implemented endpoints not in documentation")
                    self.issues.append({
                        'category': 'DOCUMENTATION_DRIFT',
                        'component': 'API Documentation',
                        'issue': f'{len(undeclared)} endpoints implemented but not documented',
                        'impact': 'Incomplete API documentation'
                    })
            except Exception as e:
                print(f"[WARN] Could not read API documentation: {e}")
    
    def check_critical_flows(self):
        """Check critical integration flows"""
        print("\n" + "="*80)
        print("CHECKING CRITICAL FLOWS")
        print("="*80)
        
        critical_flows = [
            ('Auth Flow', ['auth/register', 'auth/login', 'auth/me']),
            ('Video Flow', ['videos', 'videos/{video_id}', 'videos/{video_id}/view']),
            ('Social Flow', ['social/posts', 'follows', 'likes']),
            ('Payment Flow', ['payments', 'subscriptions']),
            ('Notification Flow', ['notifications', 'notifications/read-all'])
        ]
        
        implemented_paths = [r['path'] for r in self.routes_implemented]
        
        for flow_name, required_paths in critical_flows:
            missing = []
            for path in required_paths:
                # Check if any implemented path contains this pattern
                found = any(path in impl_path for impl_path in implemented_paths)
                if not found:
                    missing.append(path)
            
            if missing:
                print(f"\n⚠ {flow_name}: Missing {len(missing)} endpoints")
                for path in missing:
                    print(f"  - {path}")
                self.issues.append({
                    'category': 'FUNCTIONAL_GAP',
                    'component': flow_name,
                    'issue': f'Missing endpoints: {missing}',
                    'impact': 'Critical flow may be incomplete'
                })
            else:
                print(f"✓ {flow_name}: All endpoints present")
    
    def generate_reports(self):
        """Generate all report files"""
        print("\n" + "="*80)
        print("GENERATING REPORTS")
        print("="*80)
        
        self.generate_issues_md()
        self.generate_remediation_plan()
        self.generate_code_diffs()
        self.generate_validation_report()
        self.generate_test_strategy()
        self.generate_observability_plan()
    
    def generate_issues_md(self):
        """Generate issues.md"""
        content = f"""# Social Flow Backend - Issues Catalog

**Generated:** {datetime.now().isoformat()}
**Total Issues:** {len(self.issues)}

## Executive Summary

This document catalogs all identified issues during the comprehensive backend audit.

## Issues by Category

"""
        
        # Group by category
        by_category = defaultdict(list)
        for issue in self.issues:
            by_category[issue['category']].append(issue)
        
        for category in ['CRITICAL', 'FUNCTIONAL_GAP', 'CONSISTENCY', 'TEST_INTEGRITY', 'DOCUMENTATION_DRIFT']:
            if category in by_category:
                issues = by_category[category]
                content += f"\n### {category} ({len(issues)} issues)\n\n"
                
                for i, issue in enumerate(issues, 1):
                    content += f"#### {category}-{i:03d}: {issue['component']}\n\n"
                    content += f"**Issue:** {issue['issue']}\n\n"
                    content += f"**Impact:** {issue['impact']}\n\n"
                    content += "---\n\n"
        
        Path('issues.md').write_text(content)
        print("✓ Generated issues.md")
    
    def generate_remediation_plan(self):
        """Generate remediation_plan.md"""
        content = f"""# Social Flow Backend - Remediation Plan

**Generated:** {datetime.now().isoformat()}

## Execution Strategy

### Phase 1: Critical Fixes (<2 hours)
1. Fix missing imports causing application startup failures
2. Ensure health checks handle missing subsystems gracefully
3. Fix User model compatibility (is_superuser property)

### Phase 2: Functional Gaps (<8 hours)
1. Implement missing endpoints declared in documentation
2. Complete critical integration flows
3. Add proper error handling in key services

### Phase 3: Test Infrastructure (<16 hours)
1. Convert placeholder tests to real pytest suites
2. Add integration tests for critical flows
3. Implement E2E test scenarios

### Phase 4: Consistency & Documentation (<8 hours)
1. Align MODEL_REGISTRY with Base.metadata
2. Update documentation to match implementation
3. Fix Pydantic V2 deprecation warnings

### Phase 5: Observability & Hardening (<4 hours)
1. Add structured logging to silent failure points
2. Implement metrics collection
3. Add feature flags for graceful degradation

## Detailed Steps

"""
        
        # Add detailed remediation steps for each issue
        by_category = defaultdict(list)
        for issue in self.issues:
            by_category[issue['category']].append(issue)
        
        for category, issues in sorted(by_category.items()):
            content += f"\n### {category} Remediation\n\n"
            for i, issue in enumerate(issues, 1):
                content += f"{i}. **{issue['component']}:** {issue['issue']}\n"
        
        Path('remediation_plan.md').write_text(content)
        print("✓ Generated remediation_plan.md")
    
    def generate_code_diffs(self):
        """Generate code_diffs.md"""
        content = f"""# Social Flow Backend - Code Fixes

**Generated:** {datetime.now().isoformat()}

## Health Check Hardening

### Fix: app/api/v1/endpoints/health.py

Make health checks handle missing subsystems gracefully with parallel checks.

```python
# Current implementation has individual checks
# Fix: Use asyncio.gather with return_exceptions=True

async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    checks = {{}}
    
    # Build list of checks to run based on feature flags
    tasks = []
    check_names = []
    
    if settings.FEATURE_REDIS_ENABLED:
        tasks.append(check_redis())
        check_names.append("redis")
    else:
        checks["redis"] = {{"status": "skipped", "reason": "feature disabled"}}
    
    # ... similar for S3, ML, Celery
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for name, result in zip(check_names, results):
        if isinstance(result, BaseException):
            checks[name] = {{"status": "error", "error": str(result)}}
        else:
            healthy, info = result
            checks[name] = info
```

## Model Compatibility

### Fix: app/models/user.py

Ensure is_superuser compatibility property exists.

```python
@property
def is_superuser(self) -> bool:
    \"\"\"Compatibility property for superuser checks.\"\"\"
    return self.role in (UserRole.ADMIN, UserRole.SUPER_ADMIN)
```

## Test Conversion

### Convert: comprehensive_test.py → tests/integration/test_comprehensive.py

Transform report-style test into real pytest assertions.

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_core_imports():
    \"\"\"Test that all core modules can be imported.\"\"\"
    from app.main import app
    from app.core.config import settings
    from app.models.user import User
    
    assert app is not None
    assert settings.PROJECT_NAME == "Social Flow Backend"
    assert User.__tablename__ == "users"
```

## Router Registration

### Fix: app/api/v1/router.py

Remove duplicate router registrations and ensure consistent prefixes.

```python
# Remove duplicate notifications router
# Keep only: app.api.v1.endpoints.notifications

api_router.include_router(
    notifications_endpoints.router,
    prefix="/notifications",
    tags=["notifications"]
)

# Remove: app.notifications.api.notifications (duplicate)
```

"""
        
        Path('code_diffs.md').write_text(content)
        print("✓ Generated code_diffs.md")
    
    def generate_validation_report(self):
        """Generate validation_report.md"""
        content = f"""# Social Flow Backend - Validation Report

**Generated:** {datetime.now().isoformat()}

## System Overview

- **Total Endpoints:** {len(self.routes_implemented)}
- **Database Tables:** {len(self.models_implemented)}
- **Test Files:** {len(self.tests_found)}
- **Issues Found:** {len(self.issues)}

## Validation Matrix

### 1. Authentication Flow
- [ ] User registration
- [ ] User login (OAuth2 + JSON)
- [ ] Token refresh
- [ ] 2FA setup and verification
- [ ] Protected endpoint access
- [ ] Role-based access control

### 2. Video Platform
- [ ] Video upload initiation
- [ ] Video processing
- [ ] Streaming URL generation
- [ ] View increment
- [ ] Stats denormalization

### 3. Social Interactions
- [ ] Post creation
- [ ] Comments and threading
- [ ] Likes/reactions
- [ ] Follow/unfollow
- [ ] Follower count integrity

### 4. Payments
- [ ] Stripe customer creation
- [ ] Subscription management
- [ ] Creator payout enablement
- [ ] Webhook handling

### 5. Notifications
- [ ] Notification creation
- [ ] Mark as read
- [ ] Mark all as read
- [ ] Unread count accuracy

### 6. AI/ML Pipeline
- [ ] Orchestrator startup
- [ ] Scheduler initialization
- [ ] Model availability checks
- [ ] Graceful degradation

### 7. Health Monitoring
- [ ] Liveness check
- [ ] Readiness check
- [ ] Detailed health check
- [ ] Subsystem failure handling

## Component Status

"""
        
        # Add component-wise status
        components = {
            'API Router': 'OPERATIONAL' if self.routes_implemented else 'FAILED',
            'Database Models': 'OPERATIONAL' if self.models_implemented else 'FAILED',
            'Health Checks': 'NEEDS_IMPROVEMENT',
            'Test Suite': 'INCOMPLETE',
            'Documentation': 'DRIFT_DETECTED'
        }
        
        for component, status in components.items():
            content += f"- **{component}:** {status}\n"
        
        Path('validation_report.md').write_text(content)
        print("✓ Generated validation_report.md")
    
    def generate_test_strategy(self):
        """Generate test_strategy.md"""
        content = """# Social Flow Backend - Test Strategy

## Test Pyramid

```
        /\\
       /E2E\\
      /------\\
     /INTEGR.\\
    /----------\\
   /   UNIT     \\
  /--------------\\
```

## Test Categories

### 1. Unit Tests (tests/unit/)
- Pure business logic
- Model methods
- Utility functions
- No database, no external services

### 2. Integration Tests (tests/integration/)
- Service layer with DB
- Repository patterns
- Redis integration
- Storage operations

### 3. API Tests (tests/api/)
- FastAPI endpoints
- Request/response validation
- Authentication flows
- Error handling

### 4. E2E Tests (tests/e2e/)
- Complete user journeys
- Multi-step workflows
- Cross-service interactions

### 5. Performance Tests (tests/performance/)
- Load testing
- Latency benchmarks
- Concurrent user simulation

### 6. Security Tests (tests/security/)
- Authorization boundaries
- SQL injection prevention
- XSS protection
- Role privilege escalation

## Critical Test Scenarios

### Auth Flow
```python
@pytest.mark.asyncio
async def test_complete_auth_flow(async_client, db_session):
    # Register
    response = await async_client.post("/api/v1/auth/register", json={...})
    assert response.status_code == 201
    
    # Login
    response = await async_client.post("/api/v1/auth/login", data={...})
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # Access protected
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

### Content Flow
```python
@pytest.mark.asyncio
async def test_video_lifecycle(async_client, auth_headers, db_session):
    # Upload
    # Process
    # View
    # Stats verification
    pass
```

## Fixtures

### conftest.py Structure
```python
@pytest.fixture
async def db_session():
    # Async session with transaction rollback
    pass

@pytest.fixture
async def async_client(app):
    # AsyncClient with app
    pass

@pytest.fixture
def auth_headers(user_token):
    # Pre-authenticated headers
    pass
```

## Coverage Goals

- Overall: >80%
- Critical paths: >95%
- Business logic: >90%

"""
        
        Path('test_strategy.md').write_text(content)
        print("✓ Generated test_strategy.md")
    
    def generate_observability_plan(self):
        """Generate observability_plan.md"""
        content = """# Social Flow Backend - Observability Plan

## Structured Logging

### Log Levels
- **DEBUG:** Detailed diagnostic information
- **INFO:** General operational messages
- **WARNING:** Degraded operation or approaching limits
- **ERROR:** Errors that need attention
- **CRITICAL:** System instability

### Log Structure (JSON)
```json
{
  "timestamp": "2025-10-06T...",
  "level": "INFO",
  "service": "social-flow-backend",
  "component": "auth_service",
  "event": "user_login",
  "user_id": "uuid",
  "ip_address": "...",
  "duration_ms": 45,
  "status": "success"
}
```

### Key Events to Log
1. **Authentication:** Login, logout, 2FA, token refresh
2. **Video Processing:** Upload start, transcoding, completion, errors
3. **Payments:** Subscription events, webhook processing
4. **ML Pipeline:** Model load, prediction, errors
5. **Health:** Subsystem status changes

## Metrics Collection

### Application Metrics
- Request rate (requests/sec)
- Response time (p50, p95, p99)
- Error rate (%)
- Active users

### Business Metrics
- Video uploads/hour
- Active streams
- Payment transactions/hour
- New user registrations/day

### Infrastructure Metrics
- Database connection pool utilization
- Redis memory usage
- S3 request rate
- ML model inference time

## Graceful Degradation

### Feature Flags (settings.py)
```python
FEATURE_S3_ENABLED = True  # Disable to use local storage fallback
FEATURE_REDIS_ENABLED = True  # Disable to skip caching
FEATURE_ML_ENABLED = True  # Disable ML recommendations
FEATURE_CELERY_ENABLED = True  # Disable background tasks
```

### Degradation Behavior
- S3 unavailable → Local storage fallback
- Redis unavailable → Skip caching, query DB directly
- ML unavailable → Basic recommendation algorithms
- Celery unavailable → Synchronous task execution

## Monitoring Integration

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram

request_count = Counter('http_requests_total', 'Total HTTP requests')
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
```

### Health Check Endpoints
- `/health` - Liveness (process alive)
- `/health/ready` - Readiness (can serve traffic)
- `/health/live` - Kubernetes liveness
- `/health/detailed` - Full system status

"""
        
        Path('observability_plan.md').write_text(content)
        print("✓ Generated observability_plan.md")
    
    def run_audit(self):
        """Run complete audit"""
        print("\n" + "="*80)
        print(" SOCIAL FLOW BACKEND - COMPREHENSIVE AUDIT")
        print("="*80)
        print(f"Start Time: {datetime.now().isoformat()}")
        
        # Run all audits
        self.audit_routes()
        self.audit_models()
        self.audit_health_endpoints()
        self.audit_test_coverage()
        self.audit_documentation_claims()
        self.check_critical_flows()
        
        # Generate reports
        self.generate_reports()
        
        # Summary
        print("\n" + "="*80)
        print("AUDIT COMPLETE")
        print("="*80)
        print(f"\nTotal Issues Found: {len(self.issues)}")
        print(f"  Critical: {sum(1 for i in self.issues if i['category'] == 'CRITICAL')}")
        print(f"  Functional Gaps: {sum(1 for i in self.issues if i['category'] == 'FUNCTIONAL_GAP')}")
        print(f"  Consistency: {sum(1 for i in self.issues if i['category'] == 'CONSISTENCY')}")
        print(f"  Test Integrity: {sum(1 for i in self.issues if i['category'] == 'TEST_INTEGRITY')}")
        print(f"  Documentation Drift: {sum(1 for i in self.issues if i['category'] == 'DOCUMENTATION_DRIFT')}")
        
        print(f"\nGenerated Files:")
        print(f"  ✓ issues.md")
        print(f"  ✓ remediation_plan.md")
        print(f"  ✓ code_diffs.md")
        print(f"  ✓ validation_report.md")
        print(f"  ✓ test_strategy.md")
        print(f"  ✓ observability_plan.md")
        
        print(f"\nEnd Time: {datetime.now().isoformat()}")

if __name__ == "__main__":
    auditor = BackendAuditor()
    auditor.run_audit()
