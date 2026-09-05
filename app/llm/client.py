"""
Gemini LLM client setup.
Uses the new google-genai SDK (google.genai).
Reads GEMINI_API_KEY from environment — never hardcoded.
"""

from __future__ import annotations

import os

from google import genai

_client: genai.Client | None = None


def get_llm_client() -> genai.Client:
    """
    Return a singleton google.genai Client.
    Reads GEMINI_API_KEY from the environment.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Copy .env.example to .env and add your key from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_model_name() -> str:
    """Return the configured Gemini model name."""
    return os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
