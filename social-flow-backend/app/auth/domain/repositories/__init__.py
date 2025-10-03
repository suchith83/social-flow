"""
Auth Domain Repositories

Repository interfaces for the Auth bounded context.
"""

from app.auth.domain.repositories.user_repository import IUserRepository

__all__ = [
    "IUserRepository",
]
