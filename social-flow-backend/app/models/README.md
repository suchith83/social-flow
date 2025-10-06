# models

## Purpose
Central aggregation of ORM (and potentially Pydantic) models representing core domain entities.

## Key Files
| File | Role |
|------|------|
| user.py | User & profile data structures |
| social.py | Social graph & interaction models |
| video.py | Video metadata & related structures |
| ad.py | Advertisement domain models |
| payment.py | Payment & transaction models |
| notification.py | Notification entities |
| livestream.py | Live streaming entities |
| types.py | Shared model/type utilities |
| base.py | Declarative base / shared mixins |

## TODO / Roadmap
- [ ] Add model docstrings & column comments
- [ ] Introduce soft-delete mixin (if needed)
- [ ] Standardize naming & indexing strategy documentation
