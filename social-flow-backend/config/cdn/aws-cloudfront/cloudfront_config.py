"""
# Configuration models & validation
"""
# config/cdn/aws-cloudfront/cloudfront_config.py
"""
CloudFront Configuration
------------------------
Defines reusable configuration schemas for CloudFront distributions.
"""

from typing import Dict, Any


class CloudFrontDistributionConfig:
    @staticmethod
    def basic_s3_distribution(bucket_domain: str, caller_reference: str) -> Dict[str, Any]:
        """Generate minimal S3-backed CloudFront config."""
        return {
            "CallerReference": caller_reference,
            "Origins": {
                "Items": [
                    {
                        "Id": "S3Origin",
                        "DomainName": bucket_domain,
                        "S3OriginConfig": {"OriginAccessIdentity": ""},
                    }
                ],
                "Quantity": 1,
            },
            "DefaultCacheBehavior": {
                "TargetOriginId": "S3Origin",
                "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
                "ForwardedValues": {"QueryString": False, "Cookies": {"Forward": "none"}},
                "MinTTL": 0,
            },
            "Enabled": True,
            "Comment": "Basic S3 distribution",
        }

    @staticmethod
    def add_custom_error_responses(config: Dict[str, Any], error_code: int, response_page: str):
        """Add custom error response (e.g., friendly 404 page)."""
        config.setdefault("CustomErrorResponses", {"Quantity": 0, "Items": []})
        config["CustomErrorResponses"]["Items"].append(
            {"ErrorCode": error_code, "ResponsePagePath": response_page, "ResponseCode": str(error_code)}
        )
        config["CustomErrorResponses"]["Quantity"] = len(config["CustomErrorResponses"]["Items"])
        return config
