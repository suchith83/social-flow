# cache.py
import time
import pickle
import os


class Cache:
    """
    Simple in-memory and file-based caching.
    """

    def __init__(self, ttl=60, file_cache=None):
        self.ttl = ttl
        self.cache = {}
        self.file_cache = file_cache

        if file_cache and os.path.exists(file_cache):
            with open(file_cache, "rb") as f:
                self.cache = pickle.load(f)

    def get(self, key):
        if key in self.cache:
            value, expiry = self.cache[key]
            if expiry > time.time():
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = (value, time.time() + self.ttl)
        if self.file_cache:
            with open(self.file_cache, "wb") as f:
                pickle.dump(self.cache, f)
