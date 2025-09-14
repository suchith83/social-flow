# Regional load balancing exceptions
# performance/scaling/regional/exceptions.py

class RegionalError(Exception):
    """Base exception for regional scaling."""


class PolicyError(RegionalError):
    """Raised when policy decision fails."""


class HealthCheckError(RegionalError):
    """Raised when a regional health check fails."""


class DispatchError(RegionalError):
    """Raised when traffic dispatching fails."""
