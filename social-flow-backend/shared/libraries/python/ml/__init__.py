# common/libraries/python/ml/__init__.py
"""
Machine Learning Utility Library - Framework Agnostic

Features:
- Dataset handling and preprocessing
- Feature engineering (scaling, embeddings, tokenization)
- Model management (save/load/version)
- Evaluation metrics (classification, regression, ranking)
- Inference (batch + real-time)
- Experiment tracking
- Compatible with scikit-learn, PyTorch, TensorFlow
"""

__all__ = [
    "config",
    "dataset",
    "preprocessing",
    "features",
    "model_manager",
    "evaluation",
    "inference",
    "experiment",
]
