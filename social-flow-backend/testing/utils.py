"""Test helpers: import FastAPI app by string and small utilities."""

from typing import Any
import importlib


def import_app_from_string(import_str: str) -> Any:
    """
    Import an attribute using a "module.path:attribute" string.
    Example: "services.recommendation-service.src.main:app"
    """
    if ":" not in import_str:
        raise ValueError("import_str must be in form 'module.path:attribute'")
    module_path, attr = import_str.split(":", 1)
    # normalize Windows backslashes in module_path (if caller accidentally passed a filepath)
    module_path = module_path.replace("\\", ".").replace("/", ".")
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise ImportError(f"Module '{module_path}' has no attribute '{attr}'")
    return getattr(module, attr)
