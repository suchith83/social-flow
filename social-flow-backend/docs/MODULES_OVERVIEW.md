# 📦 Modules Overview

This document provides a concise, high-signal overview of the current backend module layout under `app/`. It complements `PROJECT_STRUCTURE.md` by focusing on *purpose*, *notable elements*, and *extension touch points* for each module.

> Scope: Only modules that currently exist in the repository (as of this commit). Hypothetical future folders appear only in roadmap annotations.

## 🔑 Legend
- **Layered (CA)**: Uses Clean Architecture segmentation (`domain/`, `application/`, `infrastructure/`, `presentation/`, etc.)
- **Service-Oriented**: Flat or semi-flat service + API + models structure
- **Placeholder**: Present but minimally implemented (expansion expected)

---

## 🧠 AI / ML Domains

### `ai_ml_services/`  *(Service-Oriented)*
High-level grouping façade for specialized ML capability clusters.
- Submodules: `content_moderation/`, `recommendation/`, `sentiment_analysis/`, `trending_prediction/`, `video_analysis/`
- Responsibility: Orchestrates model pipelines per capability class.
- Extension Points: Add a new capability folder; expose uniform interface contract (e.g. `.run()` or provider registry).
- Suggested TODO: Introduce a central registry & lazy loading wrapper.

### `ml/` *(Service-Oriented)*
Lower-level ML task orchestration & service endpoints.
- Key Files: `ai_processing.py`, `ml_tasks.py`
- Focus: Direct invocation of model pipelines & task scheduling hooks.
- TODO: Split experiment vs production inference paths; add model version tagging.

### `ml_pipelines/` *(Service-Oriented / Orchestration)*
Batch & scheduled lifecycle management for ML artifacts.
- Key Files: `batch_processor.py`, `orchestrator.py`, `scheduler.py`, `monitor.py`, `recommendation_precomputer.py`
- Subdirs: `data_preprocessing/`, `feature_engineering/`, `inference/`, `training/`
- TODO: Add provenance tracking & metrics export standardization.

---

## 📊 Analytics & Ads

### `analytics/`
- Components: `api/`, `routes/`, `services/`, `tasks/`, `models/`, `analytics_processing.py`
- Responsibility: Event aggregation, KPI retrieval, analytics job execution.
- TODO: Introduce dimensional modeling layer & warehouse sync abstraction.

### `ads/`
- Structure: `api/`, `models/`, `services/`
- Focus: Advertisement targeting, retrieval, attribution.
- TODO: Add bidding strategy abstraction + fraud detection pipeline hook.

---

## 🔐 Identity & Users

### `auth/` *(Layered CA)*
- Layers: `domain/`, `application/`, `infrastructure/`, `presentation/`, `api/`, `schemas/`, `services/`, `models/`
- Core Concerns: Authentication, token lifecycle, multi-factor, identity proofing.
- TODO: Unify session + device trust model; modularize rate limiting policies.

### `users/`
- Scope: User profile operations (distinct from deeper auth security concerns).
- TODO: Add profile completion scoring + public exposure policy definitions.

---

## 💬 Social / Content Domains

### `posts/` *(Layered CA)*
- Full social posting domain: creation, editing, visibility, feed integration.
- TODO: Introduce command/query segregation for write vs feed queries.

### `videos/` *(Layered CA + Processing)*
- Subdirs: `application/`, `domain/`, `infrastructure/`, `presentation/`, `routes/`, `services/`, `tasks/`
- Key Files: `video_processing.py`, `video_tasks.py`
- Pipeline: Upload → Transcode → Thumbnail / Metadata → Publish → Analytics.
- TODO: Abstract encoder backend provider & add adaptive task priority queue.

### `livestream/` *(Layered CA + Realtime)*
- Includes: `websocket/`, `routes/`, multi-layer segmentation.
- Focus: Live session lifecycle, chat, realtime metadata diffusion.
- TODO: Add viewer retention metrics & adaptive ingest scaling hooks.

### `live/`
- Appears overlapping with `livestream/` but narrower.
- TODO: Clarify intent; consider merge or rename (e.g., `live_core/`).

### `moderation/` *(Placeholder)*
- Currently minimal.
- TODO: Central policy engine + escalation workflow + audit log emitter.

### `notifications/`
- Components: `api/`, `routes/`, `services/`, `tasks/`, `models/`, `email_processing.py`, `notification_processing.py`, `websocket_handler.py`
- Channels: Email, in-app, websocket push.
- TODO: Add user preference rules engine + delivery SLA tracking.

### `payments/`
- Structure: `api/`, `models/`, `schemas/`, `services/`
- TODO: Introduce ledger abstraction + idempotent reconciliation worker.

### `analytics/` (see above) & `recommendation_service` (in `services/`) tie into feed & personalization loops.

---

## 🧩 Cross-Cutting & Shared

### `core/`
- System primitives: config, database wiring, redis, logging, security, metrics, exceptions.
- Dual variants present (`*_enhanced.py`)—suggest unifying under feature flags or strategy pattern.

### `domain/`
- Foundational domain abstractions: entities, repositories, value objects.
- TODO: Add generic Unit of Work pattern & domain event dispatcher.

### `infrastructure/`
- Storage, repository concrete implementations, persistence utilities.
- TODO: Instrument DB calls with tracing decorators.

### `shared/`
- Clean architecture style shared building blocks reused by multiple domains.
- TODO: Add docs listing which modules import each component to monitor coupling.

### `schemas/`
- Shared Pydantic models: `base.py`, `user.py`, `social.py`, `video.py`.
- TODO: Introduce versioning or compatibility layer for external clients.

### `services/`
- General-purpose service utilities (storage, search, recommendation, legacy storage path).
- TODO: Deprecate `storage_service_legacy.py` after parity validation.

### `tasks/` *(Placeholder)*
- Ready for global or shared task registrations (currently empty).

### `workers/`
- Celery bootstrap (`celery_app.py`). Consider consolidation with task orchestration modules.

---

## ⚖️ Architectural Observations
- Some domains use deep layered segmentation while others remain flat—consider standardizing for predictability.
- Overlap between `live/` and `livestream/` may confuse new contributors.
- Presence of both high-level (`ai_ml_services/`) and lower-level ML modules (`ml/`, `ml_pipelines/`) is powerful but needs a clear integration diagram.

## 🛠️ Recommended Near-Term Improvements
| Priority | Action | Rationale |
|----------|--------|-----------|
| High | Consolidate or document difference: `live/` vs `livestream/` | Reduce conceptual duplication |
| High | Add dependency flow diagram for ML subsystems | Clarify pipeline orchestration |
| Medium | Introduce CHANGELOG.md & docs index | Improve onboarding & release hygiene |
| Medium | Remove or migrate runtime artifacts (`*.db`, raw test outputs) | Cleaner repo & CI hygiene |
| Medium | Add automated module stats script (`make stats`) | Prevent documentation drift |
| Low | Unify enhanced vs base variants in `core/` | Reduce surface area |

## 🧪 Testing Touch Points
- Domain logic (layered modules) should prefer unit tests against `domain/` entities & repositories.
- Integration tests target orchestrators in `application/` or service layer.
- ML pipelines: add regression test harness around `ml_pipelines/` (deterministic sample subsets).

## 📄 Template for Per-Module README
The following template will be used when generating individual `README.md` files for each major module:

```markdown
# <Module Name>

## Purpose
Concise description.

## Responsibilities
- Bullet list of core responsibilities

## Key Directories / Files
| Path | Role |
|------|------|
| example.py | Brief explanation |

## Data & Models
Brief notes on models/entities involved.

## External Integrations
List any external systems, queues, caches, cloud resources.

## Extension Points
Patterns, abstract base classes, interfaces.

## Observability
Logging, metrics, tracing hooks.

## Security Considerations
AuthZ, data sensitivity, rate limiting, validation.

## TODO / Roadmap
- [ ] Improvement 1
- [ ] Improvement 2
```

---

If you approve, the next step will generate per-module `README.md` files using that template.
