from typing import Callable, Any, List, Dict
from .utils import get_logger

logger = get_logger(__name__)


def map_event(event: Dict, fn: Callable[[Dict], Dict]) -> Dict:
    """Applies a map transformation to an event."""
    try:
        return fn(event)
    except Exception as e:
        logger.error(f"Map operator error: {e}")
        return event


def filter_event(event: Dict, fn: Callable[[Dict], bool]) -> bool:
    """Filters events based on predicate."""
    try:
        return fn(event)
    except Exception as e:
        logger.error(f"Filter operator error: {e}")
        return False


def reduce_events(events: List[Dict], fn: Callable[[Any, Dict], Any], initializer: Any) -> Any:
    """Reduce function for aggregating a list of events."""
    result = initializer
    for e in events:
        result = fn(result, e)
    return result
