# notifications

## Purpose
Delivers user notifications across multiple channels (email, in-app, websocket) with processing & preference logic.

## Key Elements
| Path | Role |
|------|------|
| api/ | HTTP endpoints |
| routes/ | Router grouping |
| services/ | Delivery + preference logic |
| tasks/ | Background processing |
| models/ | Persistence entities |
| email_processing.py | Email pipeline workers |
| notification_processing.py | Core notification task logic |
| websocket_handler.py | Real-time push handler |

## TODO / Roadmap
- [ ] Add notification template versioning
- [ ] Implement batching & digest delivery
- [ ] Add per-channel delivery reliability metrics
