from collections import OrderedDict

class ARCCache:
    """
    Adaptive Replacement Cache (ARC) implementation.
    Balances between recency and frequency dynamically.
    """

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.T1 = OrderedDict()  # Recently used
        self.T2 = OrderedDict()  # Frequently used
        self.B1 = OrderedDict()  # Ghost entries for T1
        self.B2 = OrderedDict()  # Ghost entries for T2
        self.p = 0  # Balancing parameter

    def _replace(self, key):
        if len(self.T1) >= 1 and (len(self.T1) > self.p or (key in self.B2 and len(self.T1) == self.p)):
            old, _ = self.T1.popitem(last=False)
            self.B1[old] = None
        else:
            old, _ = self.T2.popitem(last=False)
            self.B2[old] = None

    def get(self, key):
        if key in self.T1:
            val = self.T1.pop(key)
            self.T2[key] = val
            return val
        if key in self.T2:
            val = self.T2.pop(key)
            self.T2[key] = val
            return val
        return None

    def put(self, key, value):
        if key in self.T1:
            self.T1.pop(key)
            self.T2[key] = value
            return
        if key in self.T2:
            self.T2.pop(key)
            self.T2[key] = value
            return

        if len(self.T1) + len(self.B1) == self.capacity:
            if len(self.T1) < self.capacity:
                self.B1.popitem(last=False)
                self._replace(key)
            else:
                self.T1.popitem(last=False)
        elif len(self.T1) + len(self.B1) < self.capacity and (len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2)) >= self.capacity:
            if len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) >= 2 * self.capacity:
                self.B2.popitem(last=False)
            self._replace(key)

        self.T1[key] = value

    def __repr__(self):
        return f"ARCCache(capacity={self.capacity}, size={len(self.T1)+len(self.T2)})"
