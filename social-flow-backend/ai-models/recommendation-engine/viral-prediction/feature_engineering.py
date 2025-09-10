"""
Feature engineering:
- time features: hour/day features relative to publish
- temporal shape features: early time-decay weighted counts
- network features: if graph available (user-item bipartite), compute degree, pagerank approximations
- content features: basic TF-IDF on text (if present)
- scaling and feature matrix output
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import logger

def add_time_of_day_features(df: pd.DataFrame, publish_col="publish_time"):
    df = df.copy()
    df["publish_hour"] = df[publish_col].dt.hour
    df["publish_weekday"] = df[publish_col].dt.weekday
    return df

def temporal_decay_feature(interactions: pd.DataFrame, publish_time: pd.Timestamp,
                           item_col="item_id", time_col="timestamp", decay_half_life_hours=3.0):
    """
    Compute decay-weighted sum of interactions in early window for a specific item.
    Useful if we want per-item temporal signature — here provided as an example for single item.
    """

    dt_hours = (interactions[time_col] - publish_time).dt.total_seconds() / 3600.0
    # only positive within early window
    mask = dt_hours >= 0
    dt_hours = dt_hours[mask]
    weights = np.exp(-np.log(2) * dt_hours / decay_half_life_hours)
    return weights.sum()

def compute_tfidf_on_content(meta_df: pd.DataFrame, content_col="text", max_features=2000):
    """
    Compute TF-IDF vectors for item content. Returns DataFrame of tfidf features.
    """
    if content_col not in meta_df.columns:
        logger.info("No content column for TF-IDF")
        return None
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    corpus = meta_df[content_col].fillna("").astype(str).values
    tfidf = vectorizer.fit_transform(corpus)
    feature_names = [f"tfidf_{i}" for i in range(tfidf.shape[1])]
    tfidf_df = pd.DataFrame(tfidf.toarray(), index=meta_df.index, columns=feature_names)
    return tfidf_df, vectorizer

def assemble_feature_matrix(item_df: pd.DataFrame, meta_df: pd.DataFrame = None,
                            content_col="text", scaler: StandardScaler = None):
    """
    Combine engineered features into X matrix and y label vector.
    Returns X (DataFrame), y (Series), scaler used.
    """
    df = item_df.copy()

    # basic fill
    df = df.fillna(0)

    # add time features if publish_time present
    if "publish_time" in df.columns:
        df = add_time_of_day_features(df, publish_col="publish_time")

    # if meta with content exists, compute tfidf and join
    if meta_df is not None and content_col in meta_df.columns:
        tfidf_df, vectorizer = compute_tfidf_on_content(meta_df, content_col=content_col)
        if tfidf_df is not None:
            df = df.join(tfidf_df, how="left")

    # drop columns not used as features
    drop_cols = ["viral", "horizon_count", "publish_time"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # scaler
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(feature_df.values)
    else:
        X = scaler.transform(feature_df.values)

    X_df = pd.DataFrame(X, index=feature_df.index, columns=feature_df.columns)
    y = df["viral"] if "viral" in df.columns else None
    return X_df, y, scaler
