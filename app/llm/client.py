"""
LLM client setup.
Wraps OpenAI (or compatible) client configuration.
Reads OPENAI_API_KEY from environment — never hardcoded.
"""

from __future__ import annotations

import os

from openai import OpenAI

_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """
    Return a singleton OpenAI client.
    Reads OPENAI_API_KEY from the environment.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client
