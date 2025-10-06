# domain

## Purpose
Foundational domain layer: entities, repositories, and value objects that model core business concepts independently of frameworks.

## Responsibilities
- Encapsulate business invariants
- Provide repository abstractions

## Key Elements
| Path | Role |
|------|------|
| entities/ | Domain entity definitions |
| repositories/ | Repository interfaces / base classes |
| value_objects.py | Immutable value objects |

## Extension Points
- Add domain events dispatcher
- Introduce specification pattern for complex querying

## TODO / Roadmap
- [ ] Implement Unit of Work interface
- [ ] Add domain event bus & handlers
