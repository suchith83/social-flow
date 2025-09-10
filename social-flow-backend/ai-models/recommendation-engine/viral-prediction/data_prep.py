"""
Data preparation helpers.

Assumes input interactions DataFrame with columns:
- item_id
- user_id
- event_type (e.g., view, like, share, comment)
- timestamp (pd.Timestamp)
- optional: content text, user features

Goal: for each item produce:
- early-window interactions (features)
- label: whether item becomes viral in prediction horizon

This module builds dataset-ready DataFrame for training.
"""

import pandas as pd
import numpy as np
from .config import EARLY_WINDOW, PREDICT_HORIZON, VIRAL_THRESHOLD
from .utils import logger
from datetime import timedelta

def compute_item_label(item_df: pd.DataFrame, publish_time: pd.Timestamp, baseline_count=1):
    """
    Determine whether an item becomes viral:
    Example label: (count_in_horizon / count_in_early_window) >= VIRAL_THRESHOLD
    More sophisticated options: percentile growth, absolute thresholds, or continuous regression target.
    """
    start = publish_time
    early_end = publish_time + EARLY_WINDOW
    horizon_end = publish_time + PREDICT_HORIZON

    mask_early = (item_df["timestamp"] >= start) & (item_df["timestamp"] < early_end)
    mask_horizon = (item_df["timestamp"] >= early_end) & (item_df["timestamp"] < horizon_end)

    early_count = item_df.loc[mask_early].shape[0]
    horizon_count = item_df.loc[mask_horizon].shape[0]

    # avoid division by zero
    if early_count <= 0:
        growth = horizon_count  # treat as raw counts if no early signal
    else:
        growth = horizon_count / max(early_count, baseline_count)

    viral = 1 if growth >= VIRAL_THRESHOLD else 0
    return {
        "early_count": early_count,
        "horizon_count": horizon_count,
        "growth": growth,
        "viral": viral
    }

def build_dataset(interactions: pd.DataFrame, meta: pd.DataFrame = None,
                  publish_col="publish_time", item_col="item_id", time_col="timestamp"):
    """
    Build dataset by grouping interactions by item.
    interactions: DataFrame with interaction events (at least item_id, timestamp)
    meta: DataFrame with item-level metadata, must include publish_time if not in interactions
    Returns DataFrame with features per item and label.
    """

    # ensure timestamp dtype
    interactions = interactions.copy()
    interactions[time_col] = pd.to_datetime(interactions[time_col])
    if meta is not None and publish_col in meta.columns:
        meta = meta.copy()
        meta[publish_col] = pd.to_datetime(meta[publish_col])

    records = []
    grouped = interactions.groupby(item_col)
    logger.info(f"Building dataset for {len(grouped)} items")

    for item_id, group in grouped:
        # find publish time: prefer meta, else earliest interaction
        if meta is not None and item_id in meta.index and publish_col in meta.columns:
            publish_time = meta.loc[item_id, publish_col]
        else:
            publish_time = group[time_col].min()

        # compute label and basic early/horizon stats
        stats = compute_item_label(interactions, publish_time)

        # basic features (counts, ratios)
        early_mask = (group[time_col] >= publish_time) & (group[time_col] < publish_time + EARLY_WINDOW)
        early_events = group.loc[early_mask]

        # event type breakdown
        event_counts = early_events["event_type"].value_counts().to_dict() if "event_type" in group.columns else {}
        # unique users
        unique_users = early_events["user_id"].nunique() if "user_id" in group.columns else 0
        # interactions per minute/hour in early window
        early_duration_hours = EARLY_WINDOW.total_seconds() / 3600.0
        rate_per_hour = stats["early_count"] / max(early_duration_hours, 1e-6)

        rec = {
            item_col: item_id,
            "publish_time": publish_time,
            "early_count": stats["early_count"],
            "early_unique_users": unique_users,
            "early_rate_per_hour": rate_per_hour,
            "horizon_count": stats["horizon_count"],
            "growth": stats["growth"],
            "viral": stats["viral"]
        }
        # add event counts with safe names
        for etype, cnt in event_counts.items():
            rec[f"early_event_{etype}"] = cnt

        # optional metadata merge later
        records.append(rec)

    df = pd.DataFrame.from_records(records).set_index(item_col)
    # merge metadata columns if provided
    if meta is not None:
        join_cols = [c for c in meta.columns if c != publish_col]
        if join_cols:
            df = df.join(meta[join_cols], how="left")

    logger.info(f"Built dataset with {len(df)} items")
    return df
