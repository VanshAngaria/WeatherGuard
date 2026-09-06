import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests that make real network/API calls",
    )
