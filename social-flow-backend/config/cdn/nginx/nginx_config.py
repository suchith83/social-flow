"""
# Config builders (server blocks, caching, SSL)
"""
# config/cdn/nginx/nginx_config.py
"""
Nginx Config Builder
--------------------
Programmatically build server blocks, SSL, caching.
"""

from typing import Dict


class NginxConfigBuilder:
    @staticmethod
    def server_block(domain: str, root: str, ssl: bool = False, cert: str = None, key: str = None) -> str:
        """Generate a server block config."""
        block = [
            f"server {{",
            f"    listen 80;",
            f"    server_name {domain};",
            f"    root {root};",
            f"    index index.html;",
        ]

        if ssl:
            block.append("    listen 443 ssl;")
            block.append(f"    ssl_certificate {cert};")
            block.append(f"    ssl_certificate_key {key};")
            block.append("    ssl_protocols TLSv1.2 TLSv1.3;")
            block.append("    ssl_ciphers HIGH:!aNULL:!MD5;")

        block.append("}")
        return "\n".join(block)

    @staticmethod
    def reverse_proxy(domain: str, upstream: str) -> str:
        """Build reverse proxy config."""
        return f"""
server {{
    listen 80;
    server_name {domain};

    location / {{
        proxy_pass http://{upstream};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""

    @staticmethod
    def gzip_compression() -> str:
        """Enable gzip compression for assets."""
        return """
gzip on;
gzip_types text/plain text/css application/json application/javascript application/xml+rss;
gzip_min_length 256;
"""
