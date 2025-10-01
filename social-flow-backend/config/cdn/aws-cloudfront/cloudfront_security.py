"""
# Security (WAF, HTTPS, geo-restrictions, signed URLs)
"""
# config/cdn/aws-cloudfront/cloudfront_security.py
"""
CloudFront Security
-------------------
Implements:
 - HTTPS enforcement
 - AWS WAF integration
 - Geo restrictions
 - Signed URLs & cookies
"""

import boto3
import datetime
import hmac, hashlib, base64
from typing import Dict, Any


class CloudFrontSecurity:
    def __init__(self, session=None):
        self.session = session or boto3.Session()
        self.client = self.session.client("cloudfront")

    def attach_waf(self, dist_id: str, web_acl_id: str, etag: str) -> bool:
        """Associate AWS WAF WebACL with CloudFront distribution."""
        config = self.client.get_distribution_config(Id=dist_id)
        dist_config = config["DistributionConfig"]
        dist_config["WebACLId"] = web_acl_id
        self.client.update_distribution(Id=dist_id, IfMatch=etag, DistributionConfig=dist_config)
        return True

    def enforce_https(self, config: Dict[str, Any]):
        """Ensure HTTPS-only policy for distribution."""
        config["DefaultCacheBehavior"]["ViewerProtocolPolicy"] = "https-only"
        return config

    def add_geo_restriction(self, config: Dict[str, Any], blocked_countries: list):
        """Restrict distribution access to certain geographies."""
        config["Restrictions"] = {
            "GeoRestriction": {"RestrictionType": "blacklist", "Quantity": len(blocked_countries),
                               "Items": blocked_countries}
        }
        return config

    def generate_signed_url(self, url: str, key_pair_id: str, private_key: str, expire_minutes: int) -> str:
        """Generate signed URL for restricted access."""
        expire_time = int((datetime.datetime.utcnow() + datetime.timedelta(minutes=expire_minutes)).timestamp())
        policy = f'{{"Statement":[{{"Resource":"{url}","Condition":{{"DateLessThan":{{"AWS:EpochTime":{expire_time}}}}}}}]}}'
        signature = base64.b64encode(hmac.new(private_key.encode(), policy.encode(), hashlib.sha1).digest()).decode()
        return f"{url}?Expires={expire_time}&Key-Pair-Id={key_pair_id}&Signature={signature}"
