"""Alembic environment configuration for Social Flow Backend."""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the new comprehensive models
from app.models.base import Base
from app.models import MODEL_REGISTRY

# Import all models to ensure they're registered with Base.metadata
from app.models.user import User, EmailVerificationToken, PasswordResetToken
from app.models.video import Video, VideoView
from app.models.social import Post, Comment, Like, Follow, Save
from app.models.payment import Payment, Subscription, Payout, Transaction
from app.models.ad import AdCampaign, Ad, AdImpression, AdClick
from app.models.livestream import LiveStream, StreamChat, StreamDonation, StreamViewer
from app.models.notification import Notification, NotificationSettings, PushToken

# Import settings for database URL
try:
    from app.infrastructure.config_enhanced import Settings
    settings = Settings()
except ImportError:
    # Fallback to old config if enhanced doesn't exist yet
    from app.core.config import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the SQLAlchemy URL from settings only if not already configured in alembic.ini
if not config.get_main_option("sqlalchemy.url"):
    # Use synchronous URL for Alembic (replace postgresql+asyncpg with postgresql+psycopg2)
    db_url = str(settings.database_url if hasattr(settings, 'database_url') else settings.DATABASE_URL)
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    db_url = db_url.replace("postgresql+psycopg://", "postgresql://")
    config.set_main_option("sqlalchemy.url", db_url)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Enable PostgreSQL-specific features
            compare_type=True,
            compare_server_default=True,
            # Include schemas
            include_schemas=True,
            # Render item to support enum types
            render_item=render_item,
        )

        with context.begin_transaction():
            context.run_migrations()


def render_item(type_, obj, autogen_context):
    """Render items for migration scripts."""
    # Import SQLAlchemy Enum for comparison
    from sqlalchemy import Enum as SQLEnum
    
    # Handle Enum types specially
    if type_ == "type" and isinstance(obj, SQLEnum):
        # Import the Python enum
        import importlib
        
        # Get the module and enum name
        if hasattr(obj.enum_class, '__module__'):
            module_path = obj.enum_class.__module__
            enum_name = obj.enum_class.__name__
            
            # Add import to autogen context
            autogen_context.imports.add(f"from {module_path} import {enum_name}")
            
            # Return the type string
            return f"sa.Enum({enum_name}, name='{obj.name}')"
    
    # Default rendering
    return False


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
