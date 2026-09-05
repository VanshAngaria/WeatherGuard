"""
pytest configuration.
Registers custom markers.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work
sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests as requiring network access (deselect with -m 'not live')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
