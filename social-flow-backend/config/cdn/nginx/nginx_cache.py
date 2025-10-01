"""
# Cache policies & purging
"""
# config/cdn/nginx/nginx_cache.py
"""
Nginx Cache
-----------
Manages static and reverse-proxy cache.
"""


class NginxCache:
    @staticmethod
    def static_cache() -> str:
        """Static file cache control."""
        return """
location ~* \.(jpg|jpeg|png|gif|ico|css|js|woff|woff2|ttf|svg)$ {
    expires 30d;
    access_log off;
    add_header Cache-Control "public";
}
"""

    @staticmethod
    def proxy_cache(path: str = "/var/cache/nginx") -> str:
        """Reverse proxy caching."""
        return f"""
proxy_cache_path {path} levels=1:2 keys_zone=my_cache:10m inactive=60m use_temp_path=off;

location / {
    proxy_cache my_cache;
    proxy_pass http://backend;
    proxy_cache_valid 200 302 10m;
    proxy_cache_valid 404 1m;
}
"""

    @staticmethod
    def purge_location() -> str:
        """Cache purge endpoint (secure with auth)."""
        return """
location ~ /purge(/.*) {
    proxy_cache_purge my_cache $1$is_args$args;
}
"""
