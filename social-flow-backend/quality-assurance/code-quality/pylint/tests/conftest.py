import pytest
from pylint.lint import Run
from pathlib import Path
import sys

# Ensure plugin package directory is importable for tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
