"""Adds the numbered architecture modules to Python's import path."""

from pathlib import Path
import sys


def configure_module_paths() -> None:
    root = Path(__file__).resolve().parent
    module_roots = [
        root / "02_api_backend",
        root / "03_travel_ai_agent",
        root / "04_external_data_services",
        root / "05_data_integration",
        root / "06_risk_knowledge_services",
        root / "07_decision_llm_engine",
        root / "08_recommendation_feedback",
    ]
    for module_root in reversed(module_roots):
        path = str(module_root)
        if path not in sys.path:
            sys.path.insert(0, path)
