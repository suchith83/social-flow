"""
# GDPR package initializer with metadata
"""
"""
GDPR Compliance Package Initialization
======================================

This package provides modules for ensuring compliance with the
General Data Protection Regulation (GDPR). It includes tools for:

- Handling Data Subject Access Requests (DSARs)
- Managing data deletion (Right to be Forgotten)
- Supporting consent management
- Providing data portability
- Validating GDPR requests
- Auditing compliance activity
- User notifications regarding rights

Designed for enterprise-scale distributed systems.
"""

from .gdpr_request_handler import GDPRRequestHandler
from .gdpr_policy import GDPRPolicy
from .gdpr_audit import GDPRAuditLogger
from .gdpr_data_deletion import GDPRDataDeletionService
from .gdpr_consent import GDPRConsentService
from .gdpr_portability import GDPRPortabilityService
from .gdpr_validator import GDPRValidator
from .gdpr_notifications import GDPRNotificationService
from .gdpr_exceptions import (
    GDPRRequestError,
    GDPRValidationError,
    GDPRConsentError,
    GDPRAuditError,
    GDPRPortabilityError,
)

__all__ = [
    "GDPRRequestHandler",
    "GDPRPolicy",
    "GDPRAuditLogger",
    "GDPRDataDeletionService",
    "GDPRConsentService",
    "GDPRPortabilityService",
    "GDPRValidator",
    "GDPRNotificationService",
    "GDPRRequestError",
    "GDPRValidationError",
    "GDPRConsentError",
    "GDPRAuditError",
    "GDPRPortabilityError",
]
