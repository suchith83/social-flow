# context_propagation.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Context propagation helpers.

- Sets up W3C TraceContext and Baggage propagators by default (can be extended for B3).
- Provides extract_context_from_headers / inject_context_to_headers helpers for web frameworks.
- Provides a ContextPropagator class to make using different propagators simple and testable.
"""

import logging
from typing import Dict, Iterable, Tuple, Optional

from .config import CorrelationConfig

logger = logging.getLogger("tracing.correlation.context")

try:
    from opentelemetry.propagate import get_global_textmap, set_global_textmap
    from opentelemetry.propagators.composite import CompositeHTTPPropagator
    from opentelemetry.propagators.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagators.baggage import BaggagePropagator
    # B3 is optional
    try:
        from opentelemetry.propagators.b3 import B3MultiFormat
        B3_AVAILABLE = True
    except Exception:
        B3_AVAILABLE = False
    OTel_PROP_AVAILABLE = True
except Exception:
    OTel_PROP_AVAILABLE = False
    logger.warning("OpenTelemetry propagators not available; context propagation will be a noop.")


class ContextPropagator:
    """
    Wrapper around OpenTelemetry propagators to support injection/extraction from HTTP headers.
    """

    def __init__(self, propagators: Optional[Iterable[str]] = None):
        self._propagators = propagators or CorrelationConfig.PROPAGATORS.split(",")
        if not OTel_PROP_AVAILABLE:
            self._composite = None
            return
        parts = []
        for p in self._propagators:
            p = p.strip().lower()
            if p == "tracecontext":
                parts.append(TraceContextTextMapPropagator())
            elif p == "baggage":
                parts.append(BaggagePropagator())
            elif p == "b3" and B3_AVAILABLE:
                parts.append(B3MultiFormat())
            else:
                logger.debug("Unknown or unavailable propagator: %s", p)
        # If multiple, combine into composite
        if parts:
            self._composite = CompositeHTTPPropagator(parts)
            # set as global so instrumentations respect it
            try:
                set_global_textmap(self._composite)
            except Exception:
                logger.debug("Failed to set global textmap")
        else:
            self._composite = None

    def inject(self, carrier: Dict[str, str]):
        """Inject current context into carrier (headers dict-like)."""
        if not OTel_PROP_AVAILABLE or not self._composite:
            return
        try:
            def setter(car, key, value):
                car[key] = value
            self._composite.inject(carrier, setter)
        except Exception:
            logger.exception("Context injection failed")

    def extract(self, carrier: Dict[str, str]):
        """Extract context from carrier and return context object (or None)."""
        if not OTel_PROP_AVAILABLE or not self._composite:
            return None
        try:
            def getter(car, key):
                # return list of values or None
                v = car.get(key)
                return [v] if v is not None else []
            return self._composite.extract(carrier, getter)
        except Exception:
            logger.exception("Context extraction failed")
            return None


# Convenience top-level functions
_default_propagator = ContextPropagator()


def inject_context_to_headers(headers: Dict[str, str]):
    """Inject current context into provided headers dict."""
    _default_propagator.inject(headers)


def extract_context_from_headers(headers: Dict[str, str]):
    """Extract context from headers and return an OTel context (or None)."""
    return _default_propagator.extract(headers)
