# common/libraries/python/messaging/middleware.py
"""
Middleware hooks for messaging (logging, tracing, metrics).
"""

from typing import Callable, Dict, Any

def logging_middleware(handler: Callable[[Dict[str, Any]], None]):
    def wrapper(message: Dict[str, Any]):
        print(f"[Messaging] Received: {message}")
        return handler(message)
    return wrapper

def tracing_middleware(handler: Callable[[Dict[str, Any]], None]):
    def wrapper(message: Dict[str, Any]):
        message["_trace_id"] = "trace-" + str(id(message))
        return handler(message)
    return wrapper

def metrics_middleware(handler: Callable[[Dict[str, Any]], None]):
    def wrapper(message: Dict[str, Any]):
        print(f"[Metrics] Processed message length: {len(str(message))}")
        return handler(message)
    return wrapper
