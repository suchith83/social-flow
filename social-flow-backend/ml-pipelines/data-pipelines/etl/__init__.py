# Package initializer for etl
# __init__.py
"""
ETL (Extract, Transform, Load) Package
--------------------------------------
Provides modular, reusable, and production-grade ETL pipeline components.
Includes connectors for databases, APIs, and files; flexible transformations;
and loaders for warehouses and storage.
"""

__version__ = "1.0.0"

from .extract import Extractor
from .transform import Transformer
from .load import Loader
from .pipeline import ETLPipeline
