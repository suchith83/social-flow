"""
Predictive Pipelines Package
----------------------------
This package implements reusable pipeline building blocks for:
 - preprocessing
 - training (with hyperparameter search)
 - evaluation
 - deployment (model registry + artifacts)
 - monitoring (metrics push, probes)
 - orchestration helpers and an example Airflow DAG

Design goals:
 - composable pipelines (each pipeline is a class exposing `run()` and `dry_run()`).
 - robust error handling, retries, and logging.
 - pluggable storage backends (local filesystem, S3, MLflow).
 - simple integration with CI/CD and schedulers (Airflow example included).
"""
