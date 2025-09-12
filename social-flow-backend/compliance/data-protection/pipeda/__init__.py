"""
# PIPEDA package initializer with metadata
"""
# compliance/data-protection/pipeda/__init__.py
"""
PIPEDA Compliance Package
-------------------------
Implements modules for compliance with Canada's PIPEDA:
 - Ten Fair Information Principles
 - Consent Management
 - Data Subject Rights
 - Data Handling & Safeguarding
 - Audit Logging & Breach Notification
 - API Middleware Enforcement
"""

from .pipeda_principles import PIPEDAPrinciples
from .pipeda_consent import PIPEDAConsentManager
from .pipeda_rights import PIPEDASubjectRights
from .pipeda_data_handling import PIPEDADataHandler
from .pipeda_audit import PIPEDAAuditLogger
from .pipeda_middleware import PIPEDAMiddleware
from .pipeda_utils import PIPEDAUtils
