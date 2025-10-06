# videos

## Purpose
End-to-end video content lifecycle: ingestion, processing (transcoding, thumbnails), metadata, routing, and distribution.

## Layered Structure
`application/`, `domain/`, `infrastructure/`, `presentation/`, plus `routes/`, `services/`, `tasks/` for processing.

## Key Files
| File | Role |
|------|------|
| video_processing.py | Processing orchestration logic |
| video_tasks.py | Task definitions (encoding, analysis) |

## TODO / Roadmap
- [ ] Implement adaptive queuing & task prioritization
- [ ] Add dynamic encoding ladder optimization
- [ ] Provide per-title encoding analytics export
