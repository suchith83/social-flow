# FeatureView abstraction (batch/streaming)
# feature_view.py
import pandas as pd
from typing import Dict, Any
from .utils import logger

class FeatureView:
    """
    Defines how to generate features from a source dataset.
    """

    def __init__(self, name: str, transformation_fn, entities: list, ttl: str = "7d"):
        self.name = name
        self.transformation_fn = transformation_fn
        self.entities = entities
        self.ttl = ttl

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building FeatureView: {self.name}")
        return self.transformation_fn(df)
