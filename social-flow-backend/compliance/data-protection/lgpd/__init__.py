"""
# LGPD package initializer with metadata
"""
# compliance/data-protection/lgpd/__init__.py
"""
LGPD Compliance Package
-----------------------
Implements modules to ensure compliance with Brazil's Lei Geral de Proteção de Dados (LGPD).
This package provides:
 - Data Subject Rights
 - Consent Management
 - Data Mapping & Lineage
 - Anonymization & Pseudonymization
 - Compliance Checking & Risk Scoring
 - Audit Logging & DPIA Handling
 - Middleware for API enforcement
"""

from .lgpd_rights import LGPDSubjectRights
from .lgpd_consent import LGPDConsentManager
from .lgpd_data_mapping import LGPDDataMapper
from .lgpd_compliance import LGPDComplianceChecker
from .lgpd_middleware import LGPDComplianceMiddleware
from .lgpd_anonymizer import LGPDAnonymizer
from .lgpd_audit import LGPDAuditLogger
from .lgpd_utils import LGPDUtils
