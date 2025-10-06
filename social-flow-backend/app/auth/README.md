# auth

## Purpose

Provides authentication, authorization, session, and identity management across the platform.

## Responsibilities

- Issue & validate JWT / refresh tokens
- Multi-factor + OAuth social sign-in
- User credential & session lifecycle management

## Layered Structure

| Layer | Role |
|-------|-----|
| domain/ | Core auth entities & invariants |
| application/ | Use case orchestration (login, register, revoke) |
| infrastructure/ | Persistence, external IdP adapters |
| presentation/ | (If present) view or interface adapters |
| api/ | HTTP endpoints |
| schemas/ | Pydantic request/response models |
| services/ | Shared auth-specific utilities |
| models/ | ORM models (user, session, verification) |

## Extension Points

- Plug additional OAuth providers via provider adapter pattern
- Introduce device binding / risk scoring service

## Observability

- Metrics: login success/failure, token issuance latency
- Security Logs: suspicious login attempts, MFA challenges

## Security Considerations

- Enforce strong password & rate limiting policies
- Secure refresh token rotation & revocation lists

## TODO / Roadmap

- [ ] Add WebAuthn support
- [ ] Implement adaptive authentication (risk-based)
- [ ] Consolidate session vs token revocation stores
