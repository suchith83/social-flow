# schemas

## Purpose
Shared Pydantic schema definitions for cross-module reuse (users, video, social graph, base models).

## Key Files
| File | Role |
|------|------|
| base.py | Base schema mixins/utilities |
| user.py | User-related DTOs |
| social.py | Social graph & interaction DTOs |
| video.py | Video metadata & response models |

## TODO / Roadmap
- [ ] Add backward compatibility versioning strategy
- [ ] Introduce stricter field aliasing & serialization policies
