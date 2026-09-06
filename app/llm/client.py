from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load local .env if available
_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env")
load_dotenv()

_client: genai.Client | None = None


def get_llm_client() -> genai.Client:
    """Return a singleton google.genai Client using the configured API key."""
    global _client
    api_key = os.environ.get("GEMINI_API_KEY")

    # If not in env, check Streamlit secrets or session state for cloud deployment
    if not api_key:
        try:
            import streamlit as st
            try:
                if hasattr(st, "secrets") and bool(st.secrets) and "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets.get("GEMINI_API_KEY")
                    if api_key:
                        os.environ["GEMINI_API_KEY"] = api_key
            except Exception:
                pass

            if not api_key and hasattr(st, "session_state") and st.session_state.get("gemini_api_key"):
                api_key = st.session_state.get("gemini_api_key")
                os.environ["GEMINI_API_KEY"] = api_key
        except Exception:
            pass

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please provide your key via .env, Streamlit Secrets, or the sidebar."
        )

    if _client is None or getattr(_client, "_api_key_used", None) != api_key:
        _client = genai.Client(api_key=api_key)
        _client._api_key_used = api_key
    return _client


def get_model_name() -> str:
    """Return configured Gemini model name (default: gemini-3.5-flash-lite)."""
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
