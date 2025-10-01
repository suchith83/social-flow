<!--
  Repository inventory is maintained manually during the refactor campaign.
  Update this document at the beginning of every major iteration.
-->

# Repository Inventory — Social Flow Backend

Last refreshed: 2025-10-01

The repository hosts a large-scale social media backend that mixes legacy microservices with a newer Python-first modular stack. This refresh focuses on the artifacts that will feed the modular monolith rebuild (auth, users, video, posts, live, monetization) while flagging legacy components earmarked for migration or retirement.

## Executive summary

- Primary runtime going forward is the async FastAPI app in `app/` backed by SQLAlchemy, Redis, and Celery workers. The project already includes scaffolding for logging, metrics, and service discovery, but most domain features remain unfinished.
- The `services/` folder still contains polyglot microservices (Python, Node, Go, Kotlin, Scala). With the consolidation plan, only a subset will survive as standalone components (e.g., streaming edge, ad delivery). Others are stubs or proofs of concept that can be absorbed into the modular monolith.
- Shared utilities live under `common/`; Python packages are structured but not yet published or versioned. Several modules inject `sys.path` shims and need packaging cleanup.
- Infrastructure artifacts (Docker, Terraform, deployment guides) exist but are loosely coupled to the reinvented FastAPI stack and require alignment.

## File Statistics (Updated)
- **Total Files**: ~3,500+ (approximate)
- **Python Files**: 2,522 (.py)
- **Compiled Python**: 1,034 (.pyc)
- **JavaScript**: 169 (.js)
- **TypeScript**: 118 (.ts)
- **Go**: 124 (.go)
- **Kotlin**: 59 (.kt)
- **Scala**: 10 (.scala)
- **Configuration Files**: 71 (.json), 46 (.yml), 25 (.yaml)
- **Documentation**: 116 (.md)
- **Other**: Shell scripts, Dockerfiles, Terraform, etc.

## Directory inventory

| Path | Role | Primary tech | Current maturity | Notes |
|------|------|--------------|------------------|-------|
| `app/` | Core FastAPI application (intended modular monolith) | Python 3.11, FastAPI, SQLAlchemy async | **Foundational** | Routes and services exist for most domains but many endpoints are placeholders; needs cohesive domain layering, migrations, and tests. |
| `common/` | Shared libraries for auth, database, messaging, monitoring, ML helpers | Python packages, Node modules | **Usable** | Python package has structure with `pyproject.toml`; requires packaging polish and lint/type coverage. Node/Go/Kotlin variants mostly legacy. |
| `services/` | Legacy/polyglot microservices | FastAPI, Express, Nest, Go, Kotlin, Scala | **Mixed** | Several directories are thin stubs (e.g., ads, payment). Video service (Node) is feature-rich but diverges from new design. Plan for gradual migration or API gateway façade. |
| `ai-models/`, `ml-pipelines/` | Model training/inference assets | Python (PyTorch, TF) | **Specialized** | Contain runnable training scripts and inference workers; require API adapters to integrate with the new ML service façade. |
| `analytics/`, `monitoring/`, `performance/` | Observability, analytics pipelines | Python, Spark, dashboards | **Operational** | Scripts and jobs exist, but telemetry integration with core app is incomplete; needs unified metrics/tracing story. |
| `storage/` | Storage abstractions (S3, GCS, Azure) | Python | **Usable** | Provides clients for object storage, chunked uploads, and CDN helpers; align interfaces with new video pipeline. |
| `live-streaming/`, `event-streaming/`, `edge-computing/` | Real-time ingestion and edge components | Python, Node, RTMP configs | **Prototype** | Contains RTMP ingest, WebSocket chat sketches, and edge worker examples; to be merged into live streaming module. |
| `deployment/`, `docs/`, `infra/` (under `docs`), `terraform/` | Deployment guidance and IaC blueprints | Terraform, Helm, Markdown | **Partial** | Documentation comprehensive but code samples outdated relative to new stack; treat as reference until refreshed. |
| `testing/`, `tests/`, `quality-assurance/` | Unit, integration, e2e harnesses | Pytest, Playwright, k6 | **Partial** | Scattered tests with low coverage; restructure around modular monolith services and add CI orchestration. |
| `workers/`, `scripts/`, `tools/` | Background jobs, developer tooling | Python, Node | **Mixed** | Contains Celery worker stubs and operational scripts. Requires consolidation into the new task queue architecture. |
| `src/` | Legacy NestJS gateway | TypeScript | **Deprecated** | Historical experiment; slated for removal once FastAPI gateway reaches parity. |

## Service snapshot (legacy vs. target)

| Service (legacy) | Location | Current status | Target action |
|------------------|----------|----------------|---------------|
| Ads | `services/ads-service` (FastAPI stub) | Bare skeleton with TODO | Replace with domain module in `app/ads` backed by analytics targeting and Redis caching. |
| Payments | `services/payment-service` (FastAPI stub) | Minimal Stripe placeholder | Rebuild inside modular monolith using async Stripe SDK, connected accounts, watch-time payouts. |
| Recommendations | `services/recommendation-service` (FastAPI) | Partially functional inference wrapper | Align with `ml-pipelines/` models, expose via internal gRPC/HTTP, add feedback persistence. |
| Search | `services/search-service` (FastAPI) | Basic search endpoints | Integrate with OpenSearch/Elasticsearch via async client and index management jobs. |
| Video processing | `services/video-service` (Node) + `workers/` | Mature pipeline (upload, transcode) | Port critical flows to Python async pipeline; keep Node workers temporarily for compatibility. |
| User service | `services/user-service` (Go) | Functional but isolated | Fold into core FastAPI stack; convert migrations and business logic. |
| API gateway | `services/api-gateway` (Kong plugins) | Config fragments only | New unified gateway will live in FastAPI + Envoy; keep configs as reference. |

## Shared libraries & data contracts

- `common/libraries/python` contains reusable modules for authentication, database connectivity, messaging, ML utilities, monitoring, and security. Modules are lightly typed; add `py.typed` markers, mypy configs, and packaging metadata so they can be imported without manual `sys.path` manipulation.
- `common/protobufs` and `common/schemas` define gRPC and JSON schema contracts. These must be reconciled with the new API contract before regeneration.
- `api-specs/`, `openapi.yaml`, and `postman_collection.json` are outdated. Regenerate after refactoring the FastAPI routers.

## Documentation & compliance artifacts

- Numerous reports (`*_SUMMARY.md`, `*_COMPLETE.md`) track previous automation runs. Keep for audit trail but mark superseded sections when new implementations land.
- `PROJECT_STRUCTURE.md` and `GETTING_STARTED.md` still describe the polyglot microservice layout. They need updates once the modular monolith structure is finalized.

## Risks & gaps identified

1. **Dependency drift** – multiple `requirements*.txt` files and service-level manifests are unsynchronized. Establish a single poetry/pip-tools workflow or lockfile per runtime.
2. **Authentication fragmentation** – JWT/2FA/social login exist only as design docs. No active middleware beyond `app/` scaffolding.
3. **Database schema** – Alembic project is initialized, but no consolidated migrations exist for core social/video entities. Need normalized Postgres schema as part of upcoming tasks.
4. **Observability** – Logging setup exists, Prometheus instrumentation wired, but tracing/metrics are not consistently emitted across services.
5. **Testing health** – Pytest suite currently fails (`tests/unit` run exits with status 1). Diagnose once code consolidation begins.
6. **Import Path Issues** (NEW): Models imported from `app.models.*` but located in `app/modules/*/models/`; Storage imports using `video_storage` but folders named `video-storage`; Missing dependencies like `structlog` (now installed).
7. **Service Integration Gaps** (NEW): Microservices in different languages not integrated; API gateway not configured; Inter-service communication not implemented.

## Immediate next actions

1. Run static analysis (mypy, Ruff/flake8, bandit) across `app/` and `common/`; capture deltas in `STATIC_REPORT.md`.
2. Define domain-driven package layout inside `app/` (e.g., `app/modules/{auth,users,videos,posts,live,ads,payments}`) and update import paths accordingly.
3. Produce baseline database schema & migrations for users, identity, posts, video assets, monetization, and analytics events.
4. Stabilize developer experience: refresh `docker-compose.yml` for Postgres, Redis, MinIO, Celery, and the FastAPI app; ensure `make` targets or scripts bootstrap local dev.
5. Start retiring legacy services by routing traffic through the FastAPI layer and capturing feature parity requirements in `CHANGELOG_CURSOR.md`.

## Follow-up documentation updates

- Once static analysis completes, update `STATIC_REPORT.md` with findings and remediation steps.
- Reflect structural changes and migration strategy in `CHANGELOG_CURSOR.md`, `PROJECT_STRUCTURE.md`, and `README.md` after each milestone.
- Add diffs and rationale to `CHANGESET_DIFFS/` as mandated by project guidelines.

---

Prepared by: Engineering refactor task force
