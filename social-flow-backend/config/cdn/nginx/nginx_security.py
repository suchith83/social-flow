"""
# Security rules (rate limiting, headers, WAF integration)
"""
# config/cdn/nginx/nginx_security.py
"""
Nginx Security
--------------
Add security headers, rate limiting, DoS protection.
"""


class NginxSecurity:
    @staticmethod
    def security_headers() -> str:
        """Add standard security headers."""
        return """
add_header X-Frame-Options "SAMEORIGIN";
add_header X-Content-Type-Options "nosniff";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Content-Security-Policy "default-src 'self';";
"""

    @staticmethod
    def rate_limiting() -> str:
        """Rate limiting to mitigate brute force attacks."""
        return """
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
}
"""

    @staticmethod
    def deny_bots() -> str:
        """Block bad bots via User-Agent."""
        return """
if ($http_user_agent ~* (badbot|evilbot|crawler)) {
    return 403;
}
"""
