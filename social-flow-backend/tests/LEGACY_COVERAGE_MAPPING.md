# Legacy Test Coverage Mapping (Harness → Pytest)

This document maps sections of the legacy monolithic scripts to the emerging pytest-based modular test suite, identifying remaining gaps to reach parity before removal.

## Legacy Sources

- `comprehensive_test.py`
- `advanced_integration_test.py`
- `check_integration.py`
- `test_api_endpoints.py`

## Category Mapping

| Legacy Category / Section | Description | Pytest Replacement(s) | Status |
|---------------------------|-------------|------------------------|--------|
| Core Imports (Test 1) | Module import smoke (models, services, schemas, routers) | `tests/unit/test_imports_minimal.py` (subset), `tests/unit/test_ml_service.py`, `tests/unit/test_video.py` (service init), `tests/unit/test_post_service.py`, `tests/unit/test_payment_service.py` | Partial (add CRUDBase + storage manager import asserts) |
| Configuration Tests (Test 2) | Ensures settings attrs exist & types | `tests/unit/test_settings_basics.py`, `tests/unit/test_config.py` | Covered |
| Database Models Presence (Test 3) | Checks model attributes & relationships | (Missing explicit relationship assertions) | GAP |
| API Endpoint Enumeration (Test 4) | Counts and categorizes routes | `tests/integration/test_health_endpoint.py` (health only), (No full route taxonomy yet) | Optional (Low value) |
| ML Model Predictions (Adv Test 3) | Invokes MLService methods (moderation, sentiment, tags, spam, engagement) | `tests/unit/test_ai_ml_facade.py`, `tests/unit/test_ml_service.py`, `tests/unit/test_ml_service_enhancements.py` | Covered |
| Service Layer Initialization (Adv Test 4) | Instantiation of Recommendation, Search, Video, Post, Notification services | `tests/unit/test_video.py`, `tests/unit/test_post_service.py`, `tests/unit/test_payment_service.py`, (Missing SearchService + NotificationService smoke) | Partial (add smoke) |
| Authentication Flow (Adv Test 5) | Password hashing, verification, token creation, AuthService init | `tests/unit/test_auth.py` (hashing), (Need explicit token creation & AuthService init) | Partial |
| Storage Manager (Adv Test 6) | get_storage_manager + method presence | (None) | GAP |
| ML Pipeline Orchestrator (Adv Test 7) | Orchestrator init & method existence | (None) | GAP |
| Real-world Scenario Tests (Adv Test 8) | Schema validation sequences (UserCreate, VideoCreate, PostCreate) + ML recommendation | `tests/integration/test_recommendations_flow.py` (recommendations), (Need combined scenario test) | Partial |
| AI/ML Endpoint Smoke (`test_api_endpoints.py`) | Hitting external running server endpoints via requests | Superseded by internal FastAPI test client strategy; add selective contract tests if needed | Intentional Omit (no external HTTP in unit CI) |
| Integration Environment Checks (`check_integration.py`) | Dep versions, env vars, AI model presence, endpoint count | Replace with focused unit tests + docs; endpoint count not critical | Intentional Omit |

## Planned Additions to Reach Parity (14b)

1. `tests/unit/test_model_relationships.py` – assert key SQLAlchemy relationships exist (User.videos, Post.comments, etc.).
2. `tests/unit/test_crud_base_interface.py` – reflectively verify CRUDBase exposes expected method names.
3. `tests/unit/test_storage_manager.py` – instantiate storage manager & assert method presence.
4. `tests/unit/test_search_notification_services.py` – smoke init for SearchService & NotificationService.
5. `tests/unit/test_auth_tokens.py` – create_access_token + verify_password roundtrip; AuthService init (graceful).
6. `tests/integration/test_orchestrator_pipeline.py` – orchestrator factory + method existence (execute_pipeline, get_pipeline_status) guarded by feature flags if needed.
7. `tests/integration/test_scenario_user_video_post.py` – scenario flow (UserCreate schema -> VideoCreate -> PostCreate -> recommendation facade call) without persistence (schema-level + service-level only).

## Deletions After Parity (14c)

- `comprehensive_test.py`
- `advanced_integration_test.py`
- `check_integration.py`
- `test_api_endpoints.py`

Archive summary will go into `LEGACY_TEST_SCRIPTS.md` with rationale and replacement references.

## Acceptance Criteria for Closing 14a–14d

- All GAP rows above addressed (either implemented or marked Intentional Omit with rationale).
- New tests passing in CI locally.
- Legacy scripts removed and replaced with archive doc.
- No direct external HTTP calls to a running server required for core test pass.

## Notes

- Relationship tests will avoid triggering full DB migrations by importing models only (or using an in-memory SQLite if needed later).
- Orchestrator test may be skipped if feature flag for ML pipelines is disabled (use pytest.skip with clear message).
- Scenario flow test focuses on schema validation & facade deterministic outputs to keep it fast and hermetic.
