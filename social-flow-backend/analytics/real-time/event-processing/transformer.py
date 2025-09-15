from typing import Dict, Any
from hashlib import sha256
from datetime import datetime
from .utils import get_logger

logger = get_logger(__name__)


def enrich_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Adds derived fields such as hash, normalized values, etc."""
    try:
        event["event_hash"] = sha256(f"{event['event_id']}{event['user_id']}".encode()).hexdigest()
        event["ingested_at"] = datetime.utcnow().isoformat()
        event["normalized_value"] = float(event["value"]) / 100.0  # Example normalization
        return event
    except Exception as e:
        logger.error(f"Enrichment error: {e}")
        return event
