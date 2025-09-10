"""
Feature extraction for content-based recommender.
Supports TF-IDF, Count Vectorizer, and Word2Vec embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from .utils import clean_text, logger
from .config import MAX_FEATURES, EMBEDDING_DIM


class FeatureExtractor:
    def __init__(self, method="tfidf"):
        """
        Initialize feature extractor.
        :param method: "tfidf", "count", or "word2vec"
        """
        self.method = method
        self.vectorizer = None
        self.word2vec_model = None

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        texts = texts.apply(clean_text)

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
            features = self.vectorizer.fit_transform(texts)
            logger.info("TF-IDF features extracted")
            return features.toarray()

        elif self.method == "count":
            self.vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
            features = self.vectorizer.fit_transform(texts)
            logger.info("Count features extracted")
            return features.toarray()

        elif self.method == "word2vec":
            tokenized = [t.split() for t in texts]
            self.word2vec_model = Word2Vec(sentences=tokenized, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
            features = np.array([self._get_sentence_vector(tokens) for tokens in tokenized])
            logger.info("Word2Vec features extracted")
            return features

        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def transform(self, texts: pd.Series) -> np.ndarray:
        texts = texts.apply(clean_text)

        if self.method in ["tfidf", "count"]:
            return self.vectorizer.transform(texts).toarray()
        elif self.method == "word2vec":
            tokenized = [t.split() for t in texts]
            return np.array([self._get_sentence_vector(tokens) for tokens in tokenized])

    def _get_sentence_vector(self, tokens):
        """
        Average word embeddings for sentence representation.
        """
        if not tokens:
            return np.zeros(EMBEDDING_DIM)
        vectors = [self.word2vec_model.wv[w] for w in tokens if w in self.word2vec_model.wv]
        if not vectors:
            return np.zeros(EMBEDDING_DIM)
        return np.mean(vectors, axis=0)
