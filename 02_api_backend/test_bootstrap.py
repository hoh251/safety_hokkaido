"""Shared import setup for backend tests executed from this directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import configure_module_paths

configure_module_paths()
