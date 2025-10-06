# Architecture Restructure Blueprint

This document defines the target modular architecture for the Social Flow backend. It introduces a layered structure, a lightweight dependency container, service facades, and standardized cross-cutting concerns (logging, metrics, feature flags, error handling).

## 1. Layered Structure

```
presentation (FastAPI routers, request/response models)
  ↓
application (use cases / orchestrators / service facades)
  ↓
domain (entities, value objects, domain services, enums)
  ↓
infrastructure (db models, repositories, external adapters, storage, redis, queues, ai providers)
  ↕
shared (cross-cutting utilities: config, logging, security, serialization)
```

Rules:
- Presentation must not import infrastructure directly (only via application/domain abstractions).
- Application may orchestrate multiple infrastructure adapters but owns no persistence logic.
- Domain has no knowledge of FastAPI or external libs other than typing/stdlib.
- Infrastructure never imports presentation; circular imports prohibited.

## 2. Dependency Direction

| Layer | Depends On |
|-------|------------|
| presentation | application, shared, schemas |
| application | domain, infrastructure (ports) |
| domain | shared |
| infrastructure | shared |
| shared | (stdlib / minimal third-party) |

## 3. Dependency Container (`app/application/container.py`)

Purpose: centralized, lazy, test-friendly provider of heavy dependencies (db session maker, cache, AI/ML facade, recommendation & video services). Replaces implicit module singletons with explicit accessors.

Core API:
```python
from app.application.container import get_container
container = get_container()
ml = container.ai_ml()
rec_service = container.recommendation_service(db_session)
```

Design Considerations:
- Thread-safe lazy initialization with double-checked locking for heavy objects.
- Each provider returns either a singleton (stateless) or a factory-bound instance (stateful per request) depending on semantics.
- Pluggable overrides for tests via `container.override(name, provider)`.

## 4. Service Facades

Existing: `AIServiceFacade` (new unified ML). Planned: `NotificationsFacade`, `PaymentsFacade`. Facades expose a constrained API surface and encapsulate complexity (circuit breakers, retries, fallbacks, metrics tagging).

## 5. Request Context & Correlation IDs

Middleware injects a `request_id` (UUID4) and optional `user_id` (when resolved later) into a contextvar visible to logging. Logger enriches records with request metadata.

## 6. Error Handling

Single exception translation layer converts internal exceptions to Problem Details JSON. (Planned) Remove per-module ad-hoc JSONResponse creation except where domain-specific metadata needed.

## 7. Observability Hooks

Phase 1: request latency, request id, basic structured logs.
Phase 2: per-service timing decorators (already partially in `ml_service` via `@timing`).
Phase 3: success/error counters, circuit breaker trip metrics, saturation metrics (queue depths, async tasks).

## 8. Feature Flags

Existing flags: `FEATURE_S3_ENABLED`, `FEATURE_REDIS_ENABLED`, `FEATURE_ML_ENABLED`, `FEATURE_CELERY_ENABLED`.
Planned logical groupings (future config normalization) under nested models: `StorageSettings`, `MessagingSettings`, `AISettings`.

## 9. Migration Strategy

1. Introduce container & middleware (this commit).
2. Gradually refactor services to accept dependencies via constructor but default to container fallback for backward compatibility.
3. Remove legacy global singletons after all imports updated.
4. Normalize settings & document deprecations.

## 10. Testing Strategy Alignment

Container override pattern supports:
```python
container.override("ai_ml", lambda: FakeAI())
```
Simplifies integration tests by replacing external adapters.

## 11. Removal Candidates (Planned)
- Residual imports referencing non-existent `app.ai_models.*` modules (now redundant).
- Duplicate recommendation logic once facade usage unified.

## 12. Future Enhancements
- Add lightweight async DI scope per request for short-lived objects.
- Introduce open-telemetry instrumentation wrapper.
- Add resilience policies (retry/backoff) centrally.

---
Document version: 0.1 (initial draft)
