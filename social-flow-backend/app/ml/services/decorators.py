"""Utility decorators for MLService.

Implements timing, structured logging, and safe execution wrappers.
References AI_ML_ARCHITECTURE.md (Sections: Orchestration, Monitoring & Observability).
"""
from __future__ import annotations
import time
import functools
import logging
from typing import Any, Callable, Optional, TypeVar, Dict
from app.core.exceptions import InferenceError, PipelineError

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def timing(name: Optional[str] = None):
    """Decorator capturing execution duration in milliseconds.

    Adds a '_timing' key to the returned dict (if dict) or attaches timing
    metadata in a tuple (result, meta) when non-dict to avoid breaking
    existing test expectations for legacy methods.
    """
    def decorator(func: F) -> F:
        label = name or func.__name__
        if asyncio.iscoroutinefunction(func):  # type: ignore[attr-defined]
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):  # type: ignore[misc]
                start = _now_ms()
                result = await func(*args, **kwargs)
                duration = _now_ms() - start
                if isinstance(result, dict):
                    result.setdefault("meta", {})
                    # Do not clobber existing timing keys
                    timing_meta = result["meta"].setdefault("timings", {})
                    timing_meta[label] = round(duration, 3)
                    return result
                return result, {"timings": {label: round(duration, 3)}}
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):  # type: ignore[misc]
                start = _now_ms()
                result = func(*args, **kwargs)
                duration = _now_ms() - start
                if isinstance(result, dict):
                    result.setdefault("meta", {})
                    timing_meta = result["meta"].setdefault("timings", {})
                    timing_meta[label] = round(duration, 3)
                    return result
                return result, {"timings": {label: round(duration, 3)}}
            return sync_wrapper  # type: ignore[return-value]
    return decorator


def safe_execution(error_cls=InferenceError, pipeline: bool = False):
    """Decorator to convert uncaught exceptions into ML error taxonomy.

    Args:
        error_cls: Base error class to wrap exceptions in.
        pipeline: When True, uses PipelineError for clarity.
    """
    def decorator(func: F) -> F:
        target_error = PipelineError if pipeline else error_cls
        if asyncio.iscoroutinefunction(func):  # type: ignore[attr-defined]
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):  # type: ignore[misc]
                try:
                    return await func(*args, **kwargs)
                except target_error:
                    raise
                except Exception as e:  # pragma: no cover - defensive
                    logger.exception("Unhandled ML exception in %s", func.__name__)
                    raise target_error(str(e)) from e
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):  # type: ignore[misc]
                try:
                    return func(*args, **kwargs)
                except target_error:
                    raise
                except Exception as e:  # pragma: no cover
                    logger.exception("Unhandled ML exception in %s", func.__name__)
                    raise target_error(str(e)) from e
            return sync_wrapper  # type: ignore[return-value]
    return decorator


def structured_response(wrap: bool = True, label: Optional[str] = None):
    """Standardize MLService responses.

    Keeps backward compatibility: if the wrapped function already returns
    a dict with 'success' key, it's left unchanged. If wrap=False, passes
    through untouched. Intended for NEW public methods only so existing
    tests expecting raw dicts remain valid.
    """
    def decorator(func: F) -> F:
        if not wrap:
            return func  # type: ignore[return-value]
        if asyncio.iscoroutinefunction(func):  # type: ignore[attr-defined]
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):  # type: ignore[misc]
                start = _now_ms()
                data = await func(*args, **kwargs)
                if isinstance(data, dict) and data.get("success") is not None:
                    return data
                duration = round(_now_ms() - start, 3)
                return {
                    "success": True,
                    "data": data,
                    "meta": {"label": label or func.__name__, "duration_ms": duration},
                }
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):  # type: ignore[misc]
                start = _now_ms()
                data = func(*args, **kwargs)
                if isinstance(data, dict) and data.get("success") is not None:
                    return data
                duration = round(_now_ms() - start, 3)
                return {
                    "success": True,
                    "data": data,
                    "meta": {"label": label or func.__name__, "duration_ms": duration},
                }
            return sync_wrapper  # type: ignore[return-value]
    return decorator


def build_standard_response(data: Any, label: str, duration_ms: float | None = None, **meta) -> Dict[str, Any]:
    """Helper to build standardized response dict.

    Provided so service code can manually create responses without decorator.
    """
    m = {"label": label}
    if duration_ms is not None:
        m["duration_ms"] = round(duration_ms, 3)
    m.update(meta)
    return {"success": True, "data": data, "meta": m}

# Avoid circular import issues
import asyncio  # noqa: E402  # isort:skip
