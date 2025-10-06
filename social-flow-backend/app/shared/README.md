# shared

## Purpose
Cross-domain abstractions and reusable building blocks applied by multiple feature modules.

## Structure
| Path | Role |
|------|------|
| application/ | Shared application-layer services |
| domain/ | Shared domain primitives / events |
| infrastructure/ | Shared persistence / adapter utilities |

## TODO / Roadmap
- [ ] Document which modules import each shared component
- [ ] Add dependency direction linting (enforce acyclic graph)
