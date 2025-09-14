from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache implementation."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """Retrieve item and mark as most recently used."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Insert item and evict least recently used if full."""
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __contains__(self, key):
        return key in self.cache

    def __repr__(self):
        return f"LRUCache(capacity={self.capacity}, size={len(self.cache)})"
