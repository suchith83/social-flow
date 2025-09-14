
# performance/scaling/auto_scaling/__init__.py

"""
Auto-Scaling Package
====================
This package provides a modular and extensible framework for implementing
auto-scaling strategies in distributed/cloud-native environments.

Features:
- Metric monitoring for CPU, memory, network, and custom KPIs
- Reactive scaling policies (threshold-based, scheduled, hybrid)
- Predictive scaling using machine learning models
- Simulation environment for testing scaling policies
- Orchestration engine to coordinate monitoring, policies, and execution

Usage:
    from performance.scaling.auto_scaling import Orchestrator, Config

    config = Config.load("scaling.yaml")
    orchestrator = Orchestrator(config=config)
    orchestrator.start()
"""

from .config import Config
from .orchestrator import Orchestrator
from .exceptions import AutoScalingError
