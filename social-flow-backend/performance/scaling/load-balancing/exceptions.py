# Custom exception classes
# performance/scaling/load_balancing/exceptions.py

class LoadBalancingError(Exception):
    """Base exception for load balancer."""


class AlgorithmError(LoadBalancingError):
    """Raised when algorithm fails."""


class HealthCheckError(LoadBalancingError):
    """Raised when node health check fails."""


class DispatchError(LoadBalancingError):
    """Raised when dispatching a request fails."""
