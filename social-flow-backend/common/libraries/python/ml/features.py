# common/libraries/python/ml/features.py
"""
Feature engineering utilities.
Includes text embeddings, tokenization.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

def tfidf_features(corpus: List[str], max_features: int = 5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(corpus)
    return features, vectorizer

def mean_embedding(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute mean embedding from list of vectors."""
    return np.mean(vectors, axis=0)
