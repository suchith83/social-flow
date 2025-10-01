# healthcheck.py
import socket


class HealthCheck:
    """
    Provides health & readiness probes for microservices.
    Can be used in HTTP endpoints (/healthz, /ready).
    """

    def __init__(self):
        self.checks = {}

    def add_check(self, name: str, check_func):
        """Register a health check function."""
        self.checks[name] = check_func

    def run(self):
        """Run all checks and return status."""
        results = {}
        for name, func in self.checks.items():
            try:
                results[name] = func()
            except Exception as e:
                results[name] = f"FAIL: {e}"
        return results

    @staticmethod
    def tcp_check(host: str, port: int) -> bool:
        """Check if a TCP port is reachable."""
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception:
            return False
