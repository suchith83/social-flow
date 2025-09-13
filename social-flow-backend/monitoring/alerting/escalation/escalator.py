# Base class for escalation logic
"""
Core Escalator interface and a sequential escalator implementation.

Escalator is responsible for:
- applying the escalation policy to incidents
- invoking channels (via channel package) to send notifications
- updating incident state via persistence
- handling retries/backoff
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import logging

from ..channels import BaseChannel  # type: ignore
from .escalation_policy import EscalationPolicy, EscalationLevel
from .incident import Incident, IncidentStatus
from .persistence import PersistenceAdapter, InMemoryPersistence
from .retry_backoff import RetryBackoff

logger = logging.getLogger(__name__)


class Escalator(ABC):
    """
    Abstract escalator.

    Responsibilities:
      - track where an incident is in the policy
      - invoke channel(s) to notify targets on each level
      - wait / schedule escalation to next level (here we provide synchronous behavior;
        production systems would schedule timers or use async workers)
    """

    def __init__(
        self,
        policy: EscalationPolicy,
        persistence: PersistenceAdapter = None,
        retry_backoff: Optional[RetryBackoff] = None,
    ):
        self.policy = policy
        self.persistence = persistence or InMemoryPersistence()
        self.retry_backoff = retry_backoff or RetryBackoff()

    @abstractmethod
    def escalate(self, incident: Incident, **kwargs) -> None:
        """
        Execute escalation for the given incident according to the policy.

        This method may be long-running (waiting between levels) or schedule tasks externally.
        """
        raise NotImplementedError()
