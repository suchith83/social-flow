from typing import Dict, Any
from .persistence import store_postgres, store_s3, store_elasticsearch
from .utils import get_logger

logger = get_logger(__name__)


def route_event(event: Dict[str, Any]) -> None:
    """
    Routes events to multiple sinks for durability & querying.
    """
    try:
        store_postgres(event)
        store_s3(event)
        store_elasticsearch(event)
    except Exception as e:
        logger.error(f"Routing error: {e}")
