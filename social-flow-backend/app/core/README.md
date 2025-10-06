# core

## Purpose
Hosts cross-cutting framework & platform primitives: configuration, logging, database, caching, security, metrics, exceptions.

## Key Files
| File | Role |
|------|------|
| config.py / config_enhanced.py | Settings management / environment profiles |
| database.py / database_enhanced.py | DB engine/session lifecycle |
| redis.py / redis_enhanced.py | Redis connection + cache utilities |
| logging.py / logging_config.py | Structured logging & formatting |
| security.py | Crypto helpers (hashing, JWT) |
| metrics.py | Instrumentation primitives |
| exceptions.py | Custom exception hierarchy |

## Extension Points
- Strategy pattern for enhanced vs base variants (feature toggles)
- Add tracing provider integration (OpenTelemetry)

## Observability
- Central logging config & correlation IDs
- Metric emitters (DB pool stats, cache hit rate)

## Security Considerations
- Centralize secret loading & encryption policies

## TODO / Roadmap
- [ ] Merge enhanced/* variants under configurable feature flags
- [ ] Add circuit breaker utilities
- [ ] Provide standardized retry/backoff wrapper
