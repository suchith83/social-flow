import time

class TTLCache:
    """Time-To-Live cache with expiration policy."""

    def __init__(self, capacity: int = 128, ttl: int = 60):
        self.capacity = capacity
        self.ttl = ttl
        self.data = {}

    def get(self, key):
        """Retrieve item if not expired."""
        if key in self.data:
            value, exp = self.data[key]
            if exp >= time.time():
                return value
            else:
                del self.data[key]
        return None

    def put(self, key, value):
        """Insert item with TTL, evict oldest if full."""
        if len(self.data) >= self.capacity:
            # Evict oldest expired
            expired_keys = [k for k, (_, exp) in self.data.items() if exp < time.time()]
            for k in expired_keys:
                del self.data[k]
            if len(self.data) >= self.capacity:
                self.data.pop(next(iter(self.data)))

        self.data[key] = (value, time.time() + self.ttl)

    def __repr__(self):
        return f"TTLCache(capacity={self.capacity}, ttl={self.ttl}, size={len(self.data)})"
