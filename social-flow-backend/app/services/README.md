# services

## Purpose
Shared service layer implementations used across multiple domains (storage, search, recommendation logic, legacy adapters).

## Key Files
| File | Role |
|------|------|
| storage_service.py | Primary storage abstraction |
| storage_service_legacy.py | Legacy compatibility layer |
| search_service.py | Search indexing / retrieval logic |
| recommendation_service.py | Recommendation orchestration |

## TODO / Roadmap
- [ ] Deprecate legacy storage service after parity tests
- [ ] Introduce interface contracts for service injection
- [ ] Add caching decorators for pure retrieval services
