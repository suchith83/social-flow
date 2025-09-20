#!/usr/bin/env python3
"""
Lightweight recommendation trainer for local development.

Produces a simple JSON artifact that maps popular items and a timestamped version.
"""
import argparse
import json
import os
import time
from typing import Dict, Any, List

# local imports from ml-pipelines package
from ml_pipelines.data.feature_store import InMemoryFeatureStore  # type: ignore
from ml_pipelines.model_registry import register_model  # type: ignore


def build_dummy_recommendation_model(popular_n: int = 50) -> Dict[str, Any]:
    items = [f"video_{i}" for i in range(popular_n)]
    # simple scores declining linearly
    scores = [{"item_id": it, "score": 1.0 / (idx + 1)} for idx, it in enumerate(items)]
    model = {"type": "dummy_recommender", "created_at": int(time.time()), "candidates": scores}
    return model


def save_model(model: Dict[str, Any], out_dir: str, name: str = "recommendation") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{name}_{int(time.time())}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    return path


def evaluate_model(model_path: str) -> Dict[str, Any]:
    # trivial evaluation: count candidates and average score
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    candidates = model.get("candidates", [])
    count = len(candidates)
    avg_score = sum(c.get("score", 0.0) for c in candidates) / (count or 1)
    return {"candidates": count, "avg_score": avg_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    parser.add_argument("--popular-n", type=int, default=50)
    parser.add_argument("--name", default="recommendation")
    args = parser.parse_args()

    # build and save model
    model = build_dummy_recommendation_model(popular_n=args.popular_n)
    saved = save_model(model, args.output_dir, name=args.name)
    metrics = evaluate_model(saved)

    # register in model registry
    meta = register_model(name=args.name, path=os.path.abspath(saved), metrics=metrics)

    print("Model saved to:", saved)
    print("Evaluation metrics:", metrics)
    print("Registered model id:", meta.get("id"))


if __name__ == "__main__":
    main()
