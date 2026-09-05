"""
Streamlit Frontend Entrypoint (Frontend subfolder alias).
Imports and executes the root streamlit_app.py application.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_root_app = _root / "streamlit_app.py"

if _root_app.exists():
    runpy.run_path(str(_root_app), run_name="__main__")
