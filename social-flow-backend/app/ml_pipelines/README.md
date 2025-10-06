# ml_pipelines

## Purpose
Batch & scheduled pipeline orchestration for preprocessing, feature engineering, training, and offline inference.

## Key Files
| File | Role |
|------|------|
| orchestrator.py | Central pipeline coordination |
| scheduler.py | Scheduling & cron logic |
| batch_processor.py | Handles batch batch job execution |
| recommendation_precomputer.py | Pre-computation of recommendation results |
| monitor.py | Pipeline health & performance monitoring |

## Subdirectories
- `data_preprocessing/`
- `feature_engineering/`
- `inference/`
- `training/`

## TODO / Roadmap
- [ ] Add dependency graph visualization generation
- [ ] Implement checkpoint & rollback system
- [ ] Introduce SLA tracking for each pipeline stage
