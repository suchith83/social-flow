# Repository Inventory — Social Flow Backend

This file was generated automatically during a full repository scan. It summarizes the languages, sizes (counts), top-level services, key entrypoints, and a prioritized list of modules that need attention.

Generated on: 2025-09-20

## High-level counts

- Python files: ~2848
- JavaScript/TypeScript files: ~572
- Dockerfiles: 34
- YAML/Compose/Terraform files: ~100
- Total files scanned: thousands (monorepo with modular services, infra, and tests)

## Major top-level directories (selected)

- `app/` - FastAPI application components and core business logic.
- `services/` - Multiple microservices (video-service, recommendation-service, search-service, payment-service, ads-service, view-count-service, etc.).
- `common/` - Shared libraries for Python and Node (auth, messaging, db, storage, monitoring).
- `ai-models/`, `ml-pipelines/` - Model training and inference code (recommendation, moderation, thumbnail generation, content analysis).
- `storage/` - Object-storage wrappers (S3, GCS, Azure), video storage pipelines (raw uploads, processed video, thumbnails).
- `monitoring/`, `analytics/`, `performance/` - Observability, dashboards, metrics collectors.
- `testing/`, `tests/` - Extensive unit / integration / e2e / performance test suites and CI configs.
- `infra/`, `terraform/`, `k8s/` (if present) - IaC examples and manifests.
- `scripts/`, `tools/` - Utility scripts, test harnesses, load-testing drivers.

## Key service entrypoints found

- Monorepo root API (main FastAPI app): `app/main.py`
- Node/Nest services: `src/main.ts` (NestJS app), many controllers under `src/` (videos, users, posts, auth)
- Python microservices:
  - `services/recommendation-service/src/main.py`
  - `services/search-service/src/main.py`
  - `services/payment-service/src/main.py`
  - `services/video-service/src/app.js` (Node)
  - `services/view-count-service/src/main.py`
- Workers and background processors:
  - `workers/video-processing/*` (Node processors)
  - `app/workers/*` (Python Celery workers)
  - `ml-pipelines/inference/*` (Python real-time/batch inference workers)

## Shared libraries

- `common/libraries/python` – shared Python helpers: `auth`, `database`, `messaging`, `monitoring`, `ml` (inference/training helpers)
- `common/libraries/node` – shared Node helpers for storage, video, messaging

## Important configs & infra

- `docker-compose.yml` – root compose to run core services locally.
- `openapi.yaml` – top-level OpenAPI (partial) for some services.
- `.github/workflows/ci.yml` – CI pipeline.
- `services/*/Dockerfile` – per-service Dockerfiles exist.
- Many `*.yml` files for tests/CI/k8s/chaos/load-testing.

## Testing harnesses

- `tests/`, `testing/` – contains unit, integration, e2e suites (pytest, cypress, playwright, artillery, k6, jmeter)

## Immediate issues & observations (prioritized)

1. Runtime dependency gaps in local environment — example: trying to run `services/recommendation-service` raised "No module named uvicorn". Many services expect dependencies to be installed.
2. The monorepo mixes multiple languages and frameworks (FastAPI, Node/Nest, NestJS TypeScript, Celery workers, custom workers). Integration contracts exist but some need standardization (auth tokens, API versions).
3. Several services have incomplete or missing DB migrations for their domain tables (e.g., `recommendation_feedback` referenced by the recommendation service worker).
4. Some services use ad-hoc PYTHONPATH shims to import `common` (dev-only). Convert `common/libraries/python` to an installable package or centralize with proper packaging.
5. The repository is already large and contains many production-ready pieces (monitoring, ML, infra). The work will be refactoring, standardizing, adding migration scripts, and wiring local compose for E2E tests.

## Prioritized modules that need immediate attention

1. `services/recommendation-service` — make runnable locally, add migrations for feedback, and integrate with `common` package. (High)
2. `services/video-service` (Node) — validate upload/encoding pipeline and integrate with storage and workers. (High)
3. `app/` FastAPI entrypoints — validate central app for shared semantics, auth middleware, and routes. (High)
4. `common/libraries/python` — package, type-hint, and lint. (High)
5. `storage/video-storage/raw-uploads` — ensure chunked uploads, validators, and tests. (Medium)
6. `ml-pipelines/inference/real-time` — ensure model loader/inference endpoints are callable by services. (Medium)

## Next steps (automatable)

1. Run static analysis (mypy, flake8, bandit) across Python packages and collect `STATIC_REPORT.md`.
2. Package `common/libraries/python` as an editable package and update services to use it (replace PYTHONPATH shims).
3. Add `docker-compose.dev.yml` to bring up Postgres, Redis, MinIO (S3), and core services for local E2E tests.
4. Create Alembic migrations for core schemas and deferred migrations for large changes.
5. Generate `API_CONTRACT.md` and update `openapi.yaml` based on discovered endpoints.

---

This inventory is a snapshot. Use it as the authoritative map for the next refactoring phases.
# Repository Inventory

## Overview
This repository contains a fragmented social media backend with mixed technologies (Python, Go, Node.js, TypeScript/NestJS). The current state shows incomplete implementations and disconnected modules.

## File Analysis

### Core Services (Current State)
| Service | Language | Status | Dependencies | Issues |
|---------|----------|--------|--------------|--------|
| ads-service | Python/FastAPI | Stub | None | TODO comments, no implementation |
| payment-service | Python/FastAPI | Stub | None | TODO comments, no Stripe integration |
| recommendation-service | Python/FastAPI | Partial | boto3, sagemaker | Missing uvicorn import, incomplete |
| search-service | Python/FastAPI | Partial | elasticsearch | Basic search only, no autocomplete |
| user-service | Go | Complete | Go modules | Well structured but isolated |
| video-service | Node.js | Complete | Express.js | Well structured but isolated |
| analytics-service | Scala | Complete | Spark, Kafka | Isolated, no Python integration |
| monetization-service | Kotlin | Complete | Spring Boot | Isolated, no Python integration |

### AI/ML Modules
| Module | Language | Status | Dependencies | Issues |
|--------|----------|--------|--------------|--------|
| content-analysis | Python | Complete | TensorFlow, OpenCV | Well structured, ready for integration |
| content-moderation | Python | Complete | Various ML libs | Ready for integration |
| generation | Python | Complete | Transformers, NLP | Ready for integration |
| recommendation-engine | Python | Complete | Scikit-learn, PyTorch | Ready for integration |

### Infrastructure & Config
| Component | Type | Status | Issues |
|-----------|------|--------|--------|
| Docker configs | Mixed | Partial | Inconsistent, missing main app |
| CI/CD | YAML | Complete | Well structured |
| Terraform | HCL | Partial | Some modules missing |
| Monitoring | Python | Complete | Ready for integration |

### Database & Storage
| Component | Language | Status | Issues |
|-----------|----------|--------|--------|
| database-storage | Go | Complete | Isolated from Python services |
| object-storage | Python | Complete | Ready for integration |
| video-storage | Python | Complete | Ready for integration |

## Critical Issues Identified

### 1. **Fragmented Architecture**
- Multiple languages without proper integration
- No unified API gateway
- Services communicate via direct calls (if at all)
- No shared authentication/authorization

### 2. **Incomplete Python Services**
- Most Python services are stubs with TODO comments
- No database integration
- No error handling
- No validation
- No testing

### 3. **Missing Core Features**
- No unified authentication system
- No video upload/encoding pipeline
- No real-time features (WebSockets)
- No feed generation
- No payment processing
- No notification system

### 4. **Configuration Issues**
- No centralized configuration management
- Hardcoded values in services
- No environment-specific configs
- Missing secrets management

### 5. **Testing & Quality**
- Minimal test coverage
- No integration tests
- No CI/CD for Python services
- No code quality checks

## Priority Refactoring Plan

### Phase 1: Core Python Backend (High Priority)
1. **Unified FastAPI Application**
   - Create main FastAPI app with proper structure
   - Implement shared middleware (auth, logging, rate limiting)
   - Add proper error handling and validation

2. **Database Integration**
   - PostgreSQL with SQLAlchemy/SQLModel
   - Redis for caching and sessions
   - Alembic migrations
   - Proper connection pooling

3. **Authentication System**
   - JWT with refresh tokens
   - OAuth2 integration (Google, Facebook, Twitter)
   - Role-based access control
   - Password hashing and validation

### Phase 2: Core Features (High Priority)
1. **Video Processing Pipeline**
   - Chunked upload to S3
   - Background encoding with Celery
   - HLS/DASH streaming
   - Thumbnail generation

2. **Social Features**
   - Posts, comments, likes, reposts
   - User following system
   - Feed generation with algorithms
   - Real-time notifications

3. **Monetization**
   - Ad serving system
   - Payment processing with Stripe
   - Subscription management
   - Creator payouts

### Phase 3: AI/ML Integration (Medium Priority)
1. **Content Analysis**
   - Integrate existing ML modules
   - Content moderation pipeline
   - Auto-tagging and categorization

2. **Recommendation Engine**
   - Integrate existing recommendation algorithms
   - Real-time recommendation serving
   - A/B testing framework

### Phase 4: Infrastructure (Medium Priority)
1. **DevOps & Deployment**
   - Docker containerization
   - Kubernetes manifests
   - Terraform for AWS
   - CI/CD pipelines

2. **Monitoring & Observability**
   - Structured logging
   - Metrics collection
   - Distributed tracing
   - Health checks

## Files to Delete
- `src/` (NestJS TypeScript) - Will be replaced with Python implementation
- Duplicate configuration files
- Unused test files
- Outdated documentation

## Files to Create
- `app/` - Main FastAPI application
- `app/core/` - Core functionality (auth, database, config)
- `app/api/` - API routes and endpoints
- `app/services/` - Business logic services
- `app/models/` - Database models
- `app/workers/` - Background job workers
- `app/ml/` - ML integration
- `tests/` - Comprehensive test suite
- `docker-compose.yml` - Local development
- `Dockerfile` - Production container
- `requirements.txt` - Python dependencies
- `alembic/` - Database migrations
- `scripts/` - Utility scripts

## Estimated Effort
- **Phase 1**: 2-3 days (Core backend)
- **Phase 2**: 3-4 days (Core features)
- **Phase 3**: 2-3 days (ML integration)
- **Phase 4**: 1-2 days (Infrastructure)

**Total**: 8-12 days for complete refactoring

## Next Steps
1. Create unified Python FastAPI application structure
2. Implement core authentication and database layer
3. Build video processing pipeline
4. Implement social features
5. Integrate AI/ML modules
6. Add comprehensive testing
7. Create deployment configurations
