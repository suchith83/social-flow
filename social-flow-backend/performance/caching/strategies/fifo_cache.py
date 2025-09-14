from collections import deque

class FIFOCache:
    """First In First Out cache eviction strategy."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.data = {}
        self.queue = deque()

    def get(self, key):
        """Retrieve item without affecting order."""
        return self.data.get(key)

    def put(self, key, value):
        """Insert item, evict oldest if full."""
        if key not in self.data and len(self.data) >= self.capacity:
            oldest = self.queue.popleft()
            del self.data[oldest]
        if key not in self.data:
            self.queue.append(key)
        self.data[key] = value

    def __repr__(self):
        return f"FIFOCache(capacity={self.capacity}, size={len(self.data)})"
