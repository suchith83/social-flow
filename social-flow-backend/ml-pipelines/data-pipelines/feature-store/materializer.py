# Materialize features from ETL -> store
# materializer.py
import pandas as pd
from typing import Dict
from .feature_view import FeatureView
from .storage import OfflineStore, OnlineStore
from .utils import logger, timed

class FeatureMaterializer:
    """
    Materializes features from raw data -> offline + online stores.
    """

    def __init__(self, offline: OfflineStore, online: OnlineStore):
        self.offline = offline
        self.online = online

    @timed
    def materialize(self, df: pd.DataFrame, view: FeatureView, offline_path: str, entities: str):
        features = view.build(df)

        # Save to offline
        self.offline.save_to_parquet(features, offline_path)

        # Save to online (entity_id -> row dict)
        for _, row in features.iterrows():
            eid = str(row[entities])
            self.online.write(eid, row.to_dict())
        logger.info(f"Materialized FeatureView {view.name}")
