
# performance/scaling/regional/__init__.py

"""
Regional Scaling Package
========================

This package provides abstractions for scaling and routing across multiple
geographic regions (e.g., multi-datacenter or multi-cloud deployments).

Features:
- Regional routing policies: latency-based, geo-based, weighted, failover
- Continuous health checks and latency monitoring per region
- Automatic failover when a region is unhealthy
- Config-driven orchestrator for multi-region deployments

Usage:
    from performance.scaling.regional import Orchestrator, Config

    config = Config.load("regional.yaml")
    orchestrator = Orchestrator(config)
    orchestrator.start()
"""

from .config import Config
from .orchestrator import Orchestrator
from .exceptions import RegionalError
