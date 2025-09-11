# retry.py
import time
import functools
import random


def retry(exceptions, tries=3, delay=1, backoff=2):
    """
    Retry decorator with exponential backoff.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    _tries -= 1
                    if _tries == 0:
                        raise
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator


def exponential_backoff(base=1, factor=2, jitter=True):
    """
    Exponential backoff generator.
    """
    n = 0
    while True:
        wait = base * (factor ** n)
        if jitter:
            wait += random.uniform(0, 1)
        yield wait
        n += 1
