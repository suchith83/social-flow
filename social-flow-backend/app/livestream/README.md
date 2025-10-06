# livestream

## Purpose
Implements full live streaming lifecycle: session creation, real-time transport, chat, and websocket broadcasting.

## Key Elements
| Path | Role |
|------|------|
| routes/ | HTTP endpoints for stream control |
| websocket/ | Real-time communication handlers |
| services/ | Business logic (state, metrics) |
| application/ / domain/ / infrastructure/ / presentation/ | Clean architecture segmentation |

## TODO / Roadmap
- [ ] Implement adaptive bitrate orchestration hooks
- [ ] Add viewer retention & QoS metrics
- [ ] Unify naming/overlap with `live/`
