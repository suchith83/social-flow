
# performance/cdn/edge-locations/__init__.py
"""
CDN Edge Locations Package

Provides utilities and components for managing CDN edge locations:
- registry: CRUD and metadata for edge nodes
- health_check: periodic and on-demand health checks
- geo_routing: mapping requests to nearest/optimal edge
- load_balancer: integration helpers / traffic steering
- capacity_planner: capacity estimation and scaling suggestions
- monitor_integration: hooks to push metrics/events
- utils: shared helpers (validation, backoff, typing)

Version: 1.0.0
"""
__all__ = [
    "registry",
    "health_check",
    "geo_routing",
    "load_balancer",
    "capacity_planner",
    "monitor_integration",
    "utils",
]
__version__ = "1.0.0"
