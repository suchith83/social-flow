"""Unified AI/ML service facade.

Previously this package attempted to re-export a large set of advanced model
classes from a non-existent ``app.ai_models`` tree, causing import errors and
confusion. The backend has standardized on a single orchestrating service
``MLService`` (see ``app.ml.services.ml_service``).  This package now exposes:

    get_ai_ml_service()  -> singleton MLService instance
    AIServiceFacade      -> thin proxy with stable public surface

Migration Guidelines:
---------------------
Legacy imports such as::

    from app.ml.services.ml_service import ml_service
    from app.ml.services.ml_service import MLService

SHOULD migrate to::

    from app.ai_ml_services import get_ai_ml_service
    ml = get_ai_ml_service()

During the transition period the old imports still work, but emit a
``DeprecationWarning`` (see patch appended to ``ml_service.py``). This allows a
gradual refactor while providing a single canonical access path for
instrumentation, feature flags, and future dependency injection.

Rationale:
----------
* Eliminates broken import surfaces referencing absent ``app.ai_models``.
* Centralizes lazy model initialization + capability registry.
* Provides seam for metrics / tracing wrappers without touching call sites.
* Eases unit testing by enabling monkeypatch of a single getter function.

NOTE: Advanced model classes (YOLO/Whisper/CLIP/etc.) remain internally
handled by ``MLService`` with safe fallbacks when unavailable.
"""

from __future__ import annotations

from typing import Any
import threading

try:  # Local import to avoid heavy imports at interpreter start
    from app.ml.services.ml_service import MLService, ml_service as _legacy_singleton
except Exception:  # pragma: no cover - extremely defensive
    MLService = Any  # type: ignore
    _legacy_singleton = None  # type: ignore

_singleton_lock = threading.Lock()
_facade_singleton: AIServiceFacade | None = None


class AIServiceFacade:
    """Proxy facade exposing a curated, stable subset of MLService methods.

    Attribute access falls through to the underlying ``MLService`` instance so
    existing method names remain usable. This façade is the indirection layer
    where we can later add:
      * metrics / tracing decorators
      * circuit breakers / timeout wrappers
      * permission / quota enforcement
    without editing every call site across the backend.
    """

    def __init__(self, core_service: MLService):  # type: ignore[name-defined]
        self._core = core_service

    # Explicitly surfaced common methods (documentation + type hints future)
    def __getattr__(self, item: str):  # pragma: no cover - passthrough
        return getattr(self._core, item)

    @property
    def capabilities(self):  # noqa: D401
        """Expose underlying capability registry (read-only)."""
        return self._core.capabilities


def _build_singleton() -> AIServiceFacade:
    # Prefer existing global instance created inside ml_service.py to avoid
    # duplicating model initialization overhead.
    core = _legacy_singleton if _legacy_singleton is not None else MLService()
    return AIServiceFacade(core)


def get_ai_ml_service() -> AIServiceFacade:
    """Return process-wide AI/ML facade singleton.

    Thread-safe lazy initialization. Subsequent calls are cheap.
    """
    global _facade_singleton
    if _facade_singleton is None:
        with _singleton_lock:
            if _facade_singleton is None:  # double-checked locking
                _facade_singleton = _build_singleton()
    return _facade_singleton


__all__ = [
    "AIServiceFacade",
    "get_ai_ml_service",
]

