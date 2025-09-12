"""
# Handle Data Subject Access Requests (DSAR)
"""
"""
CCPA Request Handler
--------------------
Coordinates validation, processing, auditing, and notification
for CCPA consumer requests.
"""

from datetime import datetime
from .ccpa_validator import CCPAValidator
from .ccpa_audit import CCPAAuditLogger
from .ccpa_data_deletion import CCPADataDeletionService
from .ccpa_optout import CCPAOptOutService
from .ccpa_notifications import CCPANotificationService
from .ccpa_exceptions import CCPARequestError

class CCPARequestHandler:
    """Main orchestrator for handling CCPA requests."""

    def __init__(self, database: dict, audit_log="ccpa_audit.log"):
        self.validator = CCPAValidator()
        self.audit = CCPAAuditLogger(audit_log)
        self.deletion_service = CCPADataDeletionService(database)
        self.optout_service = CCPAOptOutService()
        self.notifier = CCPANotificationService()

    def process_request(self, request: dict, request_type: str, user_email: str):
        """
        Process a CCPA request end-to-end.
        """
        # 1. Validate
        self.validator.validate_request(request, request_type)

        # 2. Audit
        self.audit.log_request(request["request_id"], request["user_id"], request_type)

        # 3. Process
        response = None
        if request_type == "deletion":
            response = self.deletion_service.delete_user_data(request["user_id"])
        elif request_type == "optout":
            response = self.optout_service.opt_out(request["user_id"])
        elif request_type == "access":
            # Placeholder: would normally query DB
            response = {"status": "success", "data": {"example": "user data"}}
        else:
            raise CCPARequestError(f"Unsupported request type: {request_type}")

        # 4. Notify user
        subject = f"CCPA {request_type.title()} Request Processed"
        message = f"Your {request_type} request (ID: {request['request_id']}) has been processed.\n\nDetails: {response}"
        self.notifier.send_notification(user_email, subject, message)

        return response
