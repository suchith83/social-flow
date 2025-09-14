import heapq
from collections import defaultdict

class LFUCache:
    """Least Frequently Used cache with eviction policy."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.data = {}
        self.freq = defaultdict(int)
        self.min_heap = []

    def get(self, key):
        """Retrieve item and update frequency."""
        if key not in self.data:
            return None
        self.freq[key] += 1
        heapq.heappush(self.min_heap, (self.freq[key], key))
        return self.data[key]

    def put(self, key, value):
        """Insert item and evict least frequently used if full."""
        if self.capacity == 0:
            return
        if key in self.data:
            self.data[key] = value
            self.get(key)  # update frequency
            return

        if len(self.data) >= self.capacity:
            # Evict LFU item
            while self.min_heap:
                freq, k = heapq.heappop(self.min_heap)
                if self.freq[k] == freq:
                    del self.data[k]
                    del self.freq[k]
                    break

        self.data[key] = value
        self.freq[key] = 1
        heapq.heappush(self.min_heap, (1, key))

    def __repr__(self):
        return f"LFUCache(capacity={self.capacity}, size={len(self.data)})"
