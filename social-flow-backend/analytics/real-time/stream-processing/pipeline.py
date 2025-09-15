import asyncio
from .processor import StreamProcessor
from .utils import get_logger

logger = get_logger(__name__)


def transform_fn(event: dict) -> dict:
    """Example transform: normalize value field."""
    if "value" in event:
        event["value"] = float(event["value"]) / 100.0
    return event


def predicate_fn(event: dict) -> bool:
    """Keep only purchase events."""
    return event.get("event_type") == "purchase"


def aggregator_fn(acc: dict, event: dict) -> dict:
    """Aggregate purchase values into running total."""
    acc["total_purchases"] = acc.get("total_purchases", 0) + event.get("value", 0)
    return acc


async def main():
    processor = StreamProcessor(transform_fn, predicate_fn, aggregator_fn)
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())
