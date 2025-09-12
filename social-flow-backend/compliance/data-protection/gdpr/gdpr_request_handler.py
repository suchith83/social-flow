"""
# Handle GDPR subject access & rights requests
"""
"""
GDPR Request Handler
--------------------
Coordinates validation, processing, auditing, and notification.
"""

from .gdpr_validator import GDPRValidator
from .gdpr_audit import GDPRAuditLogger
from .gdpr_data_deletion import GDPRDataDeletionService
from .gdpr_consent import GDPRConsentService
from .gdpr_portability import GDPRPortabilityService
from .gdpr_notifications import GDPRNotificationService
from .gdpr_exceptions import GDPRRequestError

class GDPRRequestHandler:
    """Main orchestrator for GDPR requests."""

    def __init__(self, database: dict, audit_log="gdpr_audit.log"):
        self.validator = GDPRValidator()
        self.audit = GDPRAuditLogger(audit_log)
        self.deletion_service = GDPRDataDeletionService(database)
        self.consent_service = GDPRConsentService()
        self.portability_service = GDPRPortabilityService(database)
        self.notifier = GDPRNotificationService()
        self.database = database

    def process_request(self, request: dict, request_type: str, user_email: str):
        # 1. Validate
        self.validator.validate_request(request, request_type)

        # 2. Audit
        self.audit.log_request(request["request_id"], request["user_id"], request_type)

        # 3. Process
        response = None
        if request_type == "deletion":
            response = self.deletion_service.delete_user_data(request["user_id"])
        elif request_type == "consent":
            response = self.consent_service.give_consent(request["user_id"])
        elif request_type == "portability":
            file_path = f"{request['user_id']}_export.json"
            self.portability_service.export_data(request["user_id"], file_path)
            response = {"status": "success", "file": file_path}
        elif request_type == "access":
            response = {"status": "success", "data": self.database.get(request["user_id"], {})}
        else:
            raise GDPRRequestError(f"Unsupported request type: {request_type}")

        # 4. Notify
        subject = f"GDPR {request_type.title()} Request Processed"
        message = f"Your {request_type} request (ID: {request['request_id']}) has been processed.\n\nDetails: {response}"
        self.notifier.send_notification(user_email, subject, message)

        return response
