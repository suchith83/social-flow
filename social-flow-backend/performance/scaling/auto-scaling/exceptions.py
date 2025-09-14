# Custom exception classes
# performance/scaling/auto_scaling/exceptions.py

class AutoScalingError(Exception):
    """Base exception for auto-scaling system."""


class PolicyError(AutoScalingError):
    """Raised when a scaling policy fails."""


class MonitoringError(AutoScalingError):
    """Raised when monitoring fails."""


class ExecutionError(AutoScalingError):
    """Raised when scaling execution fails."""
