# profiler.py
import cProfile
import pstats
import io
from typing import Callable


class Profiler:
    """
    Lightweight performance profiler.
    Wraps around Python's cProfile.
    """

    def __init__(self):
        self.profiler = cProfile.Profile()

    def start(self):
        self.profiler.enable()

    def stop(self):
        self.profiler.disable()

    def report(self, sort_by="cumulative", top=10) -> str:
        """Return profiling stats as a string."""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(top)
        return s.getvalue()

    def profile_function(self, func: Callable):
        """Decorator to profile a function."""
        def wrapper(*args, **kwargs):
            self.start()
            result = func(*args, **kwargs)
            self.stop()
            print(self.report())
            return result
        return wrapper
