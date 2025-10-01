"""
# Common helpers (ARN utils, ID generation)
"""
# config/cdn/aws-cloudfront/cloudfront_utils.py
"""
CloudFront Utilities
--------------------
Common helpers for ARN parsing, ID generation, etc.
"""

import uuid


class CloudFrontUtils:
    @staticmethod
    def generate_caller_reference() -> str:
        """Generate unique caller reference for distribution creation."""
        return str(uuid.uuid4())

    @staticmethod
    def parse_distribution_arn(arn: str) -> str:
        """Extract distribution ID from ARN."""
        return arn.split("/")[-1]

    @staticmethod
    def sanitize_domain(domain: str) -> str:
        """Ensure domain has no protocol prefix."""
        return domain.replace("https://", "").replace("http://", "")
