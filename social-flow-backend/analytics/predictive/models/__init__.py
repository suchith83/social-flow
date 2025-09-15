"""
Predictive Models Package
-------------------------
Provides data loading, feature engineering, model training, evaluation,
explainability, inference, and registry utilities for predictive analytics.

Modules:
 - config.py: Configuration (pydantic)
 - data_loader.py: Load data from warehouse / Parquet / feature store
 - feature_engineering.py: Feature transforms and pipelines
 - model_trainer.py: Train and persist models (supports CV, hyperparam search)
 - model_registry.py: Register, version and retrieve models (local + MLflow optional)
 - inference.py: Batch and online inference helpers
 - evaluation.py: Metrics and evaluation pipeline
 - explainability.py: SHAP-based explainability helpers
 - utils.py: Logging, serialization, decorators
"""
