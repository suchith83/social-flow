import asyncio
import json
from .stream_reader import read_from_kafka
from .schema import Event
from .transformer import enrich_event
from .filter import filter_event
from .router import route_event
from .utils import get_logger

logger = get_logger(__name__)


async def run_pipeline():
    """Main orchestrator of event processing pipeline."""
    async for raw_event in read_from_kafka():
        try:
            parsed = json.loads(raw_event)
            event = Event(**parsed).dict()  # Validate against schema
            enriched = enrich_event(event)
            if filter_event(enriched):
                route_event(enriched)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
