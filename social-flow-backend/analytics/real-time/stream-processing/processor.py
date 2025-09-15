import asyncio
import json
from typing import Callable
from .window import SlidingWindow
from .operators import map_event, filter_event, reduce_events
from .state import StateStore
from .sinks import publish_to_kafka
from .sources import consume_kafka
from .utils import get_logger

logger = get_logger(__name__)


class StreamProcessor:
    """Core stream processor for transformations and windowed aggregations."""

    def __init__(self, transform: Callable, predicate: Callable, aggregator: Callable):
        self.window = SlidingWindow()
        self.state = StateStore()
        self.transform = transform
        self.predicate = predicate
        self.aggregator = aggregator

    async def run(self):
        """Run continuous stream processing loop."""
        async for raw_event in consume_kafka():
            try:
                event = json.loads(raw_event)
                mapped = map_event(event, self.transform)
                if filter_event(mapped, self.predicate):
                    self.window.add(mapped)
                    events = self.window.get_events()
                    agg_result = reduce_events(events, self.aggregator, initializer={})
                    self.state.update("latest_agg", agg_result)
                    await publish_to_kafka(json.dumps(agg_result))
            except Exception as e:
                logger.error(f"StreamProcessor error: {e}")
