# env.py
import os


class Env:
    """
    Helper for environment variable management.
    """

    @staticmethod
    def get(key: str, default=None, cast_type=str):
        value = os.getenv(key, default)
        if value is not None and cast_type:
            try:
                return cast_type(value)
            except Exception:
                return default
        return value

    @staticmethod
    def require(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"Missing required env var: {key}")
        return value
