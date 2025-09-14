
# performance/scaling/load_balancing/__init__.py

"""
Load Balancing Package
======================

Provides a modular and extensible framework for implementing
software-defined load balancing strategies.

Features:
- Multiple algorithms: round robin, least connections, weighted, hashing
- Health checks to detect and isolate failing nodes
- Session affinity (sticky sessions) with consistent hashing
- Monitoring of node utilization
- Orchestration layer to combine monitoring, algorithms, and request dispatch

Usage:
    from performance.scaling.load_balancing import Orchestrator, Config

    config = Config.load("lb.yaml")
    orchestrator = Orchestrator(config)
    orchestrator.start()
"""

from .config import Config
from .orchestrator import Orchestrator
from .exceptions import LoadBalancingError
