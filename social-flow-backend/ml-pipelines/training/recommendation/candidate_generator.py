# First stage candidate retrieval (ANN, embeddings, etc.)
"""
candidate_generator.py
----------------------
Implements first-stage candidate retrieval using embeddings + ANN search (FAISS).
"""

import numpy as np
import faiss
from typing import List, Dict


class CandidateGenerator:
    """
    Candidate generator based on user/item embeddings.
    Uses FAISS for efficient Approximate Nearest Neighbor search.
    """

    def __init__(self, embedding_dim: int, num_candidates: int = 100):
        self.embedding_dim = embedding_dim
        self.num_candidates = num_candidates
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim if normalized)
        self.item_embeddings = None

    def fit(self, item_embeddings: np.ndarray):
        """Builds FAISS index for item embeddings."""
        self.item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        self.index.add(self.item_embeddings.astype(np.float32))

    def generate(self, user_embedding: np.ndarray) -> List[int]:
        """Retrieve top-K candidates for a given user embedding."""
        user_embedding = user_embedding / np.linalg.norm(user_embedding)
        D, I = self.index.search(np.array([user_embedding], dtype=np.float32), self.num_candidates)
        return I[0].tolist()


if __name__ == "__main__":
    np.random.seed(42)
    items = np.random.rand(1000, 64).astype(np.float32)
    generator = CandidateGenerator(embedding_dim=64, num_candidates=10)
    generator.fit(items)
    user_vec = np.random.rand(64)
    print("Top candidates:", generator.generate(user_vec))
