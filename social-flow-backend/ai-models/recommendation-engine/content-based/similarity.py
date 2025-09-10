"""
Similarity computation for content-based recommendations.
Supports cosine, dot product, and Euclidean similarity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import SIMILARITY_METRIC


class SimilarityComputer:
    def __init__(self, metric: str = SIMILARITY_METRIC):
        self.metric = metric

    def compute(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Compute similarity scores.
        :param matrix: Item feature matrix
        :param vector: Query vector
        """
        if self.metric == "cosine":
            return cosine_similarity(matrix, vector.reshape(1, -1)).flatten()
        elif self.metric == "dot":
            return np.dot(matrix, vector)
        elif self.metric == "euclidean":
            return -np.linalg.norm(matrix - vector, axis=1)  # Negative because lower distance = higher similarity
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")
