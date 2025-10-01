"""
CCPA Compliance Package Initialization
======================================

This package provides modules for ensuring compliance with the 
California Consumer Privacy Act (CCPA). It includes tools for:

- Handling data subject access requests (DSARs)
- Managing data deletion requests
- Supporting "Do Not Sell My Personal Information" opt-outs
- Validating CCPA request formats
- Auditing compliance activity
- User notifications regarding their rights

Each module is designed to integrate with enterprise-scale platforms.
"""

from .ccpa_request_handler import CCPARequestHandler
from .ccpa_policy import CCPAPolicy
from .ccpa_audit import CCPAAuditLogger
from .ccpa_data_deletion import CCPADataDeletionService
from .ccpa_optout import CCPAOptOutService
from .ccpa_validator import CCPAValidator
from .ccpa_notifications import CCPANotificationService
from .ccpa_exceptions import (
    CCPARequestError,
    CCPAValidationError,
    CCPANotificationError,
    CCPAAuditError,
)

__all__ = [
    "CCPARequestHandler",
    "CCPAPolicy",
    "CCPAAuditLogger",
    "CCPADataDeletionService",
    "CCPAOptOutService",
    "CCPAValidator",
    "CCPANotificationService",
    "CCPARequestError",
    "CCPAValidationError",
    "CCPANotificationError",
    "CCPAAuditError",
]
