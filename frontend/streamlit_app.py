"""
Streamlit Frontend — Weather Advisory Support Bot
Minimal chat UI with conversational history and session/thread ID.
"""

import os
import sys
import uuid

import streamlit as st

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

from app.graph.graph import run_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Weather Advisory Support Bot",
    page_icon="⛅",
    layout="centered",
)

st.title("⛅ Weather Advisory Support Bot")
st.caption(
    "Ask me weather-safety questions like: *'Is it safe to cycle in Mumbai today?'* "
    "or *'Should I take my child to the park in Delhi?'*"
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Display thread ID (for debugging / multi-session)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Session Info")
    st.code(st.session_state.thread_id, language=None)
    st.caption("Each browser session has a unique thread ID. Refresh to start a new session.")

    if st.button("🔄 New Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This bot uses:\n"
        "- **Open-Meteo** for live weather data\n"
        "- **Deterministic SOPs** for safety decisions\n"
        "- **LangGraph** for orchestration\n"
        "- **LLM** for language understanding only"
    )

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a weather-safety question…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run graph
    with st.chat_message("assistant"):
        with st.spinner("Checking weather and safety policies…"):
            try:
                answer = run_graph(
                    user_message=prompt,
                    thread_id=st.session_state.thread_id,
                )
            except Exception as exc:
                answer = (
                    f"❌ **Unexpected Error**\n\n"
                    f"Something went wrong: {exc}\n\n"
                    f"Please ensure your OPENAI_API_KEY is set in your `.env` file."
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
