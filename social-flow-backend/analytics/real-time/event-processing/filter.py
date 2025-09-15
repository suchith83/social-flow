from typing import Dict, Any
from .utils import get_logger

logger = get_logger(__name__)


def filter_event(event: Dict[str, Any]) -> bool:
    """
    Apply filtering rules.
    Returns True if event should be processed further.
    """
    try:
        if event.get("value", 0) < 0:
            logger.debug(f"Filtered negative value: {event}")
            return False
        if event.get("event_type") not in {"click", "view", "purchase"}:
            logger.debug(f"Filtered unknown type: {event}")
            return False
        return True
    except Exception as e:
        logger.error(f"Filtering error: {e}")
        return False
