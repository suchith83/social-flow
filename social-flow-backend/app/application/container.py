"""Lightweight dependency container for the Social Flow backend.

The container centralizes construction of heavy or shared dependencies while
remaining framework-agnostic. It provides:
  * Lazy singleton providers for stateless services
  * Factory helpers for request-scoped / db-bound services
  * Override capability for tests (simple monkeypatch pattern)

Design goals: minimal surface, explicit over implicit, no magic meta-programming.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import threading
import uuid

from app.core import database
from app.core.redis import get_cache, get_redis  # type: ignore

try:
    from app.ai_ml_services import get_ai_ml_service
except Exception:  # pragma: no cover
    def get_ai_ml_service():  # type: ignore
        return None

from app.videos.services.video_service import VideoService  # type: ignore
from app.services.recommendation_service import RecommendationService  # type: ignore


class Container:
    """Central dependency registry.

    Access via module-level `get_container()` to maintain a single instance.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._singletons: Dict[str, Any] = {}
        self._overrides: Dict[str, Callable[[], Any]] = {}

    # --- Override API (primarily for tests) ---
    def override(self, name: str, provider: Callable[[], Any]) -> None:
        with self._lock:
            self._overrides[name] = provider
            if name in self._singletons:
                self._singletons.pop(name, None)

    def clear_override(self, name: str) -> None:
        with self._lock:
            self._overrides.pop(name, None)

    # --- Internal helper ---
    def _resolve(self, name: str, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if name in self._overrides:
                return self._overrides[name]()
            if name not in self._singletons:
                self._singletons[name] = factory()
            return self._singletons[name]

    # --- Providers ---
    def ai_ml(self):  # facade singleton
        return self._resolve("ai_ml", get_ai_ml_service)

    def redis(self):  # async lazy – returns coroutine
        async def _redis():
            return await get_redis()
        return _redis()

    def cache(self):  # async lazy – returns coroutine
        async def _cache():
            return await get_cache()
        return _cache()

    def db_session_factory(self):  # returns the async sessionmaker
        return database.get_session_maker()

    def video_service(self):  # stateless wrapper singleton
        def _factory():
            return VideoService()
        return self._resolve("video_service", _factory)

    def recommendation_service(self, db_session=None):  # new instance per explicit db for clarity
        if db_session is None:
            # Provide on-demand ephemeral session (test convenience); prefer explicit injection in real endpoints
            Session = database.get_session_maker()
            return RecommendationService(db=Session())  # type: ignore[arg-type]
        return RecommendationService(db=db_session)

    def request_id(self) -> str:
        return uuid.uuid4().hex


_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
    return _container


__all__ = ["get_container", "Container"]
