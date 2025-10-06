# Social Flow Backend - Issues Catalog# Issue Catalog



**Generated:** 2025-10-06  Comprehensive categorized issue catalog for Social Flow Backend audit.

**Audit Type:** Comprehensive Deep Audit  

**Total Issues:** 37## Legend

- Category: [Critical] breaks runtime or causes undefined behavior; [Functional Gap] missing required documented function; [Consistency] naming or duplication mismatch; [Performance]; [Security]; [Test Integrity]; [Documentation Drift].

## Executive Summary- ID format: CAT-###.

- Status: Open (default), Quick-Win Candidate (QW), Needs Design (ND), Deferred (DEF).

This comprehensive audit analyzed the Social Flow Backend across authentication, videos, social interactions, payments, notifications, AI/ML pipelines, storage, and health monitoring. The audit identified critical gaps, consistency issues, test integrity problems, and documentation drift.

---

**Key Findings:**

- 273 API endpoints implemented (exceeds documented 107+)## 1. Critical Issues

- 31 database models registered

- 55 test files found, but 3 are placeholder/report-style tests| ID | Title | Description | Impact | Suggested Action | Status |

- Multiple missing AI/ML model modules (`app.ai_models`)|----|-------|-------------|--------|------------------|--------|

- Pydantic V1→V2 deprecation warnings need addressing| CRT-001 | Duplicate Router Mounts (Notifications) | Both `app.api.v1.endpoints.notifications` and `app.notifications.api.notifications` mounted (same prefix `/notifications`) causing potential path shadowing. | Unpredictable routing, wrong handler execution | Remove one mount; keep consolidated newer endpoints; adjust docs | Open |

- Health checks lack full graceful degradation| CRT-002 | Duplicate Router Mounts (Analytics) | Two analytics routers (`analytics` and `analytics_enhanced`) on `/analytics`. | Colliding paths; last include wins silently | Namespace second router or merge | Open |

- Notification endpoint naming inconsistency| CRT-003 | Health Endpoint Duplication | Root `/health` (in `main.py`) and versioned `/api/v1/health/*` create ambiguity; docs cite only 3 endpoints. | Monitoring confusion & inconsistent tooling | Decide canonical pattern; root simple liveness + versioned detailed | Open |

| CRT-004 | Missing Video View Increment Endpoint | Docs define `POST /videos/{id}/view` but absent. | Client features relying on view count updates break | Implement atomic counter endpoint | Open |

---| CRT-005 | Admin User Management Endpoints Missing | Activation / suspension / role management absent though referenced in docs. | Moderation controls impossible through API | Implement or adjust docs; minimal 501 placeholders first | Open |

| CRT-006 | Optional Subsystem Failures Hard-Fail Health | S3/Redis/ML failures mark readiness unhealthy without feature gating. | Causes false downtime & auto restarts | Add feature flags + degrade status | Open |

## Issues by Category| CRT-007 | Legacy + v2 Namespace Unscoped | `/v2/*` coexists with legacy without version negotiation; docs ignore them. | Client confusion & accidental use of unstable endpoints | Add explicit documentation + maybe experimental tag | Open |

| CRT-008 | Potential Migration / Model Drift (User & Stripe) | Large user model fields; need verification vs latest migrations (e.g. restructure migration 006). | Runtime errors if missing columns in deployed DB | Perform migration diff audit | Open |

### CRITICAL (6 issues)| CRT-009 | Router Order & Tag Overlaps | Mixed capitalization tags (Payments vs payments) and duplicates risk inconsistent OpenAPI generation. | Tooling / SDK generation inconsistency | Normalize tags; rerun openapi spec | Open |

| CRT-010 | AI Pipeline Endpoint Naming Divergence | Docs: `/pipelines/run`, code: `/pipelines/tasks`; mismatch may break client integrations. | 404 for clients following docs | Add alias or update docs | Open |

#### CRIT-001: Missing AI Models Package

**Component:** AI/ML Services  ## 2. Functional Gaps

**Issue:** Module `app.ai_models` not found. Multiple imports fail:

- `app.ai_models.content_moderation`| ID | Title | Description | Impact | Action | Status |

- `app.ai_models.recommendation`|----|-------|-------------|--------|--------|--------|

- `app.ai_models.video_analysis`| FNC-011 | Push Token Management Endpoints Missing | Docs mention push notifications; no endpoints for token register/delete. | Cannot deliver multi-device push | Add CRUD or mark deferred | Open |

- `app.ai_models.sentiment_analysis`| FNC-012 | Email Verification Flow Not Implemented | Only TODO comment; docs promise flow. | Users cannot verify emails | Add verification token endpoints & status flag | Open |

- `app.ai_models.trending_prediction`| FNC-013 | Password Breach Detection Absent | Promised in README security claims. | Reduced security posture | Integrate HaveIBeenPwned or local compromised password list (optional flag) | Deferred |

| FNC-014 | OAuth Provider Logins Absent | Docs list Google/Twitter/etc; not implemented. | Reduced onboarding pathways | Add provider auth or downgrade docs | Deferred |

**Impact:** AI/ML functionality completely unavailable, recommendation engine cannot load models  | FNC-015 | Monetization Gating Endpoint | No explicit `can_monetize` evaluation route. | Clients must replicate logic | Add lightweight read-only endpoint | Open |

**Severity:** CRITICAL  | FNC-016 | Recommendation Warm Cache Endpoint Missing | Docs list `warm-cache`. | Precompute strategies not triggerable via API | Implement orchestrator call | Open |

**Remediation Time:** 4-8 hours| FNC-017 | Scheduler Start/Stop Endpoints Unverified | Docs specify start/stop; not yet confirmed present. | Ops cannot manage scheduler remotely | Implement endpoints or doc removal | Open |

| FNC-018 | Live Streaming Interactive Features | Polls/Q&A/donations not surfaced. | Feature incompleteness | Mark roadmap in docs | Deferred |

#### CRIT-002: Missing Analytics Package| FNC-019 | Grouped Notifications Endpoint Missing | Grouping claimed; not implemented. | UI cannot group efficiently | Add grouping or clarify doc removal | Deferred |

**Component:** Analytics Module  | FNC-020 | Content Moderation AI Endpoints Missing | Models listed (NSFW, toxicity, etc.) no public endpoints. | Cannot externally leverage moderation service | Add endpoints or note internal-only | Deferred |

**Issue:** Import errors for analytics modules not available

## 3. Consistency Issues

**Impact:** Analytics features non-functional  

**Severity:** CRITICAL  | ID | Title | Description | Impact | Action |

**Remediation Time:** 2-4 hours|----|-------|-------------|--------|--------|

| CNS-021 | Endpoint Naming: task vs pipeline | Discrepancy between docs and `ai_pipelines.py`. | Developer confusion | Choose canonical terms; update docs & optional alias |

#### CRIT-003: Missing python-jose Dependency| CNS-022 | Mixed Tag Capitalization | `Payments` vs `payments`. | OpenAPI consumers get duplicates | Normalize tags to lowercase |

**Component:** Dependencies  | CNS-023 | Field Naming: `qr_code_url` vs doc naming | Must ensure schema & docs aligned. | Minor friction | Sync doc wording |

**Issue:** Required package `python-jose` not installed| CNS-024 | Health Status Values | Code uses `healthy/degraded/not_ready` but docs simplified. | Dashboard misinterpretation | Document states & add legend |

| CNS-025 | Social visibility enums | Docs say `followers_only`; posts use `followers`; need unification. | Access control edge bugs | Standardize & add migration if needed |

**Impact:** Authentication may fail  | CNS-026 | Video visibility values | Ensure doc set matches enum (`followers_only` vs code usage). | Filtering inconsistencies | Align constants and docs |

**Severity:** CRITICAL  | CNS-027 | Duplicate analytics endpoints | Enhanced vs base not documented difference. | Overwrite risk | Merge or version new analytics |

**Remediation Time:** 5 minutes| CNS-028 | Legacy vs v2 docs absence | `/v2/*` not documented. | Unofficial use risk | Add dedicated section |



#### CRIT-004: Database URL Configuration Missing## 4. Performance / Scalability

**Component:** Configuration  

**Issue:** DATABASE_URL not set in environment| ID | Title | Description | Impact | Action |

|----|-------|-------------|--------|--------|

**Impact:** Application cannot connect to database  | PRF-029 | Lack of Caching Layer in Recommendations Fallback | Repeated DB heavy queries. | Latency under load | Add Redis caching w/ TTL & feature flag |

**Severity:** CRITICAL  | PRF-030 | Trending Queries Counting Each Request | Uses length of returned list not total count query. | Imprecise pagination & potential N+ usage | Add true count query |

**Remediation Time:** 10 minutes| PRF-031 | Missing DB Index Verification | Denormalized counters rely on frequent updates; need composite indexes (e.g. created_at + owner_id). | Slow queries on scale | Audit & add migrations |

| PRF-032 | Health Detailed Check Serial Latency | Some checks could time out; already parallel but Celery/ML calls might slow. | Slower readiness | Timeout wrappers + classify optional |

#### CRIT-005: Health Check Exception Handling| PRF-033 | No Rate Limiting Layer | README references architecture with rate limiting; absent. | Abuse risk | Add middleware or gateway note |

**Component:** Health Endpoints  

**Issue:** May not gracefully handle missing subsystems## 5. Security Issues



**Impact:** Health endpoints may return 500 errors  | ID | Title | Description | Impact | Action |

**Severity:** HIGH  |----|-------|-------------|--------|--------|

**Remediation Time:** 1 hour| SEC-034 | Missing Email Verification Enforcement | Users default to `PENDING_VERIFICATION` but flows unenforced. | Account trust gap | Enforce active status gating certain actions |

| SEC-035 | Suspension/Ban Checks Not Centralized | Endpoints do not uniformly call a can_post/can_interact guard. | Suspended accounts may interact | Implement permission helpers & integrate |

#### CRIT-006: Unicode/Encoding Issues| SEC-036 | 2FA Backup Code Usage Missing | Backup codes generated but no consumption endpoint. | Reduced account recovery | Add endpoint & one-time use logic |

**Component:** Test Infrastructure  | SEC-037 | Stripe Webhook Processing Visibility | Webhook events table created; need idempotency & signature validation auditing. | Financial risk | Verify webhook handler; add tests |

**Issue:** UTF-8 characters not compatible with Windows console| SEC-038 | Admin Privilege Escalation Safeguards | Need central decorator verifying admin & optional super admin. | Role confusion | Consolidate permission decorators |

| SEC-039 | Password Breach / Strength Enforcement | Only length & character checks described; no runtime breach check. | Weak credential acceptance | Add optional HIBP integration |

**Impact:** Test execution fails on Windows  

**Severity:** MEDIUM  ## 6. Test Integrity Issues

**Remediation Time:** 30 minutes

| ID | Title | Description | Impact | Action |

---|----|-------|-------------|--------|--------|

| TST-040 | Harness Scripts Not Pytest-Assertive | `comprehensive_test.py` etc rely on prints/logging. | False confidence | Convert to modular pytest tests |

### FUNCTIONAL_GAP (8 issues)| TST-041 | Missing E2E Auth → Content Flow Tests | No combined scenario test. | Regressions unnoticed | Add scenario test with fixtures |

| TST-042 | No Health Degradation Test | Optional subsystem failure not simulated. | Fragile production behavior | Add redis-down test using monkeypatch |

#### FUNC-001: Notification Endpoint Naming Inconsistency| TST-043 | No Role/Permission Matrix Tests | Admin vs user paths untested. | Security regressions | Parametrized tests over roles |

**Component:** Notifications API  | TST-044 | No Stripe Connect Flow Test | Migrations & models unvalidated at runtime. | Financial feature drift | Mock stripe & test connect lifecycle |

**Issue:** Documentation mentions `mark-all-read` but implementation uses `read-all`| TST-045 | Recommendation Algorithm Fallback Tests Missing | ML unavailable path not covered. | Runtime errors uncaught | Simulate ADVANCED_ML_AVAILABLE False |

| TST-046 | Concurrency Counter Integrity | View/like increments not concurrency-tested. | Data skew | Use async tasks & assert counts |

**Impact:** API consumer confusion  

**Severity:** MEDIUM  ## 7. Documentation Drift

**Remediation Time:** 15 minutes

| ID | Title | Description | Action |

---|----|-------|-------------|--------|

| DOC-047 | README Endpoint Count Inflation | 107+ likely includes duplicates & future endpoints. | Recount automatically & update |

### CONSISTENCY (7 issues)| DOC-048 | AI Model Inventory vs Implementation | Many models listed without endpoints. | Add INTERNAL ONLY note or implement shells |

| DOC-049 | Live Streaming Feature Claims | Advanced interaction claims missing. | Create roadmap section |

#### CONS-001: Pydantic V1→V2 Deprecation Warnings| DOC-050 | Performance Metrics (50-200ms inference) | Marketing numbers not backed by benchmarks. | Add TBD metrics or remove |

**Component:** Schemas  | DOC-051 | Roadmap File References Missing | Broken `PHASE_7_8_TESTING_GUIDE.md`. | Remove or create stub |

**Issue:** Multiple files using deprecated Pydantic V1 `@validator`| DOC-052 | Notification Grouping & Channels | Channels referenced; implementation partial. | Clarify scope |

| DOC-053 | Rate Limiting Layer Shown in Diagram | No implementation. | Annotate diagram (planned) |

**Impact:** Future Pydantic V3 incompatibility  | DOC-054 | Export Capabilities (CSV/JSON) | Not implemented. | Mark planned |

**Severity:** MEDIUM  

**Remediation Time:** 2 hours## 8. Observability Gaps



---| ID | Title | Description | Action |

|----|-------|-------------|--------|

### TEST_INTEGRITY (3 issues)| OBS-055 | Lack of Structured Metrics for Startup Phases | No timers for DB, Redis, Orchestrator init. | Add timing instrumentation |

| OBS-056 | Health Check Log Context Minimal | Failures log only error strings. | Add structured fields (component, latency_ms) |

#### TEST-001: Placeholder Test: comprehensive_test.py| OBS-057 | Missing Correlation/Request IDs | No middleware adding trace IDs. | Implement simple UUID header propagation |

**Component:** Test Suite  | OBS-058 | Recommendation Cache Hit Metrics | Not tracked; just returns cached flag. | Add counter/gauge |

**Issue:** Report-style script, not real pytest assertions

## 9. Data Integrity / Model Issues

**Impact:** False sense of test coverage  

**Severity:** HIGH  | ID | Title | Description | Action |

**Remediation Time:** 4 hours|----|-------|-------------|--------|

| DAT-059 | Soft Delete Approach Inconsistent | Videos mark status DELETED; posts use hard delete. | Standardize approach or doc rationale |

---| DAT-060 | Denormalized Counters Race Risk | Follower/like counts can drift under concurrency without DB constraints. | Use atomic SQL updates or optimistic locking |

| DAT-061 | Missing Unique Constraints Validation | Need audit (e.g. combination constraints in social interactions). | Review and add migrations |

### DOCUMENTATION_DRIFT (5 issues)| DAT-062 | Cascade Policies Mixed | Some FKs CASCADE, others SET NULL without domain rationale. | Document or align policies |

| DAT-063 | Missing Partition Strategy for High-Volume Tables | Future partition hints not implemented. | Plan migration strategy |

#### DOC-001: Endpoint Count Discrepancy

**Component:** README  ## 10. Deferred / Strategic Enhancements

**Issue:** README claims "107+ endpoints" but 273 routes found

| ID | Title | Description | Action |

**Impact:** Documentation understates capabilities  |----|-------|-------------|--------|

**Severity:** LOW  | DEF-064 | Feature Flag Framework | Wrap optional subsystems behind toggles. | Implement after core fixes |

**Remediation Time:** 10 minutes| DEF-065 | Automated Endpoint Inventory Script | Generate JSON for docs sync. | Add CLI script |

| DEF-066 | OpenAPI Linter Pipeline | Prevent drift & duplicates. | Integrate spectral or similar |

---| DEF-067 | Changelog Automation | Track notable changes for releases. | Introduce conventional commits + tool |

| DEF-068 | Load/Latency Benchmark Suite | Back performance claims with data. | Add locust / k6 smoke tests |

### SECURITY (4 issues)

---

#### SEC-001: No Privilege Escalation Tests

**Component:** Authorization  ## Cross-Cutting Root Causes

**Issue:** No tests verifying users cannot escalate roles1. Documentation ahead of implementation (product vision embedded inside README/API doc).

2. Partial refactor to v2 endpoints without deprecating legacy or documenting dual-phase.

**Impact:** Potential security vulnerability  3. Lack of feature flags forcing optional services to appear mandatory.

**Severity:** HIGH  4. Test harnesses optimized for reporting rather than enforcement.

**Remediation Time:** 2 hours5. Observability not embedded early — limited feedback loops for reliability & performance.



---## Immediate Quick Win Set (Top 10)

1. Remove duplicate router mounts (CRT-001, CRT-002).

## Issue Statistics2. Implement /videos/{id}/view (CRT-004).

3. Add feature flags & degrade health logic (CRT-006, DEF-064 start).

- **Total Issues:** 374. Provide admin user management 501 stubs (CRT-005) with consistent responses.

- **Critical:** 6 (16%)5. Create endpoint inventory script stub (DEF-065) for future automation (fast scaffold).

- **High Severity:** 7 (19%)6. Normalize tags and remove capitalization mismatches (CRT-009, CNS-022).

- **Medium Severity:** 16 (43%)7. Add correlation-id middleware (OBS-057) & structured logging upgrades (OBS-055/056).

- **Low Severity:** 8 (22%)8. Convert one harness script to real pytest test (TST-040 seed work).

9. Add `status: degraded` semantics & document health states (CNS-024).

---10. Add alias or documentation fix for pipeline tasks vs run (CRT-010, CNS-021).



**See remediation_plan.md for detailed fix strategy**## Blocking Dependencies (Sequencing Pointers)

- Health hardening depends on feature flags (CRT-006 ↔ DEF-064).
- Accurate remediation plan depends on migration audit (CRT-008 → DAT-059..062).
- Test suite uplift should occur after endpoint stubs to avoid churn (TST-040 after CRT-005 stubs).

## Acceptance Criteria for Remediation Completion
- OpenAPI spec no duplicate or shadowed paths; tags normalized.
- Health checks distinguish required vs optional subsystems with feature flags.
- All documented critical endpoints either implemented or explicitly marked 501 with roadmap note.
- Pytest suite includes: auth flow, content flow (post or video + like), notification mark-all-read, health degraded scenario.
- Issue catalog updated with statuses and linked PR references.

---

End of Issue Catalog.
