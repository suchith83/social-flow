# Retrieve point-in-time correct features
# retriever.py
import pandas as pd
from typing import Dict, Any, List
from .storage import OfflineStore, OnlineStore
from .utils import logger, timed

class FeatureRetriever:
    """
    Retrieves features for training (offline) or inference (online).
    """

    def __init__(self, offline: OfflineStore, online: OnlineStore):
        self.offline = offline
        self.online = online

    @timed
    def get_online(self, entity_ids: List[str]) -> Dict[str, Any]:
        results = {}
        for eid in entity_ids:
            results[eid] = self.online.read(eid)
        return results

    @timed
    def get_offline(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        logger.info(f"Retrieved {len(df)} rows from offline store {path}")
        return df
