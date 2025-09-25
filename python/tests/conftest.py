# tests/conftest.py
import sys
from pathlib import Path

# Add project root to sys.path so "apps", "solver", "vision" can be imported in tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
