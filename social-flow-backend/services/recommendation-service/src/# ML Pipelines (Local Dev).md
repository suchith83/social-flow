# ML Pipelines (Local Dev)

This folder contains lightweight, dependency-free helpers for model training, registry and a minimal serving endpoint useful for local development and CI smoke-tests.

Quick examples:
- Train a dummy recommendation model:
  python ml-pipelines/training/recommendation/train.py --output-dir ml-pipelines/models

- Run a minimal prediction server (serves predictions from latest model):
  python ml-pipelines/inference/real_time/serve.py --model-path ml-pipelines/models/<model.json> --port 8600

Notes:
- Artifacts are simple JSON files; replace training logic with real frameworks when integrating production models.
- The model registry is a JSON file `ml-pipelines/registry.json` that records model metadata.
