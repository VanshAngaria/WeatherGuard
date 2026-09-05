"""
Streamlit Frontend — Weather Advisory Support Bot
Professional UI with example prompts, pipeline explainer,
scope section, and live session context display.
"""

from __future__ import annotations

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
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional, clean, decision-support feel
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Import professional font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0f1117;
}

/* Hide default Streamlit header padding */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 860px;
}

/* Header card */
.header-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2d4a 50%, #152038 100%);
    border: 1px solid #2a3f6f;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
}

.header-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #e8eef8;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.3px;
}

.header-subtitle {
    font-size: 0.95rem;
    color: #7b9ec7;
    margin: 0 0 0.6rem 0;
    font-weight: 400;
}

.header-desc {
    font-size: 0.85rem;
    color: #5c7da8;
    margin: 0;
}

/* Center Help Card — What I Can Help With */
.help-card {
    background: #121829;
    border: 1px solid #233354;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.3rem;
}

.help-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #1c2740;
    padding-bottom: 0.5rem;
}

.help-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2edfa;
    letter-spacing: -0.2px;
}

.help-subtitle {
    font-size: 0.77rem;
    color: #799ec2;
    font-style: italic;
}

.help-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
}

@media (max-width: 768px) {
    .help-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.help-chip {
    background: #172136;
    border: 1px solid #293a5c;
    border-radius: 6px;
    padding: 0.5rem 0.65rem;
    font-size: 0.79rem;
    color: #b0c9e8;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.15s ease;
}

.help-chip:hover {
    background: #202e4d;
    border-color: #3b5380;
    color: #ffffff;
}

/* Sidebar Pipeline Stepper */
.sidebar-pipeline {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin: 0.6rem 0;
}

.pipeline-step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
}

.pipeline-step-badge {
    background: #1b2742;
    color: #72a4e2;
    border: 1px solid #2d4570;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.68rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}

.pipeline-step-text {
    font-size: 0.78rem;
    color: #b0cceb;
    line-height: 1.3;
}

.pipeline-step-text small {
    color: #5c7ba1;
    font-size: 0.69rem;
    display: block;
    margin-top: 1px;
}

.pipeline-step-connector {
    color: #273b5c;
    font-size: 0.75rem;
    padding-left: 0.42rem;
    line-height: 1;
    margin: 3px 0;
}

/* Example prompts section */
.examples-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #5c7da8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.6rem;
}

/* Example button styling */
.stButton > button {
    background: #141824 !important;
    border: 1px solid #2a3550 !important;
    color: #a8c4e0 !important;
    border-radius: 8px !important;
    padding: 0.45rem 0.7rem !important;
    font-size: 0.8rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    line-height: 1.4 !important;
}

.stButton > button:hover {
    background: #1e2d4a !important;
    border-color: #3d5a8a !important;
    color: #cce0f5 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2d45 !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #7b9ec7;
}

/* Session badge */
.session-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0d1f0d;
    border: 1px solid #1a3d1a;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #4caf50;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

.session-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4caf50;
    display: inline-block;
}

/* Context info box */
.context-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.6rem 0;
}

.context-row {
    font-size: 0.78rem;
    color: #7b9ec7;
    padding: 2px 0;
}

.context-key {
    color: #5c7da8;
    font-weight: 500;
}

.context-val {
    color: #a8c4e0;
    font-weight: 600;
}

/* Chat area */
.stChatMessage {
    border-radius: 10px !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid #1e2d45 !important;
    margin: 1rem 0 !important;
}

/* Scope section */
.scope-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
}

.scope-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #5c7da8;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 0.5rem;
}

.scope-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 16px;
}

.scope-item {
    font-size: 0.79rem;
    color: #7b9ec7;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_context" not in st.session_state:
    st.session_state.session_context = {"location": None, "activity": None, "time": None}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Session status
    st.markdown(
        '<div class="session-badge"><span class="session-dot"></span> Session Active</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Session Context")
    st.caption(
        "This session remembers your location, activity, and time so "
        "follow-up questions work naturally."
    )

    # Context display
    ctx = st.session_state.session_context
    loc = ctx.get("location") or "—"
    act = ctx.get("activity") or "—"
    tim = ctx.get("time") or "—"

    st.markdown(f"""
<div class="context-box">
  <div class="context-row"><span class="context-key">📍 Location &nbsp;</span><span class="context-val">{loc}</span></div>
  <div class="context-row"><span class="context-key">🚴 Activity &nbsp;</span><span class="context-val">{act}</span></div>
  <div class="context-row"><span class="context-key">🕐 Time &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="context-val">{tim}</span></div>
</div>
""", unsafe_allow_html=True)

    st.caption(
        "**Example follow-up flow:**  \n"
        "\"Is it safe to cycle in Bhopal?\" → \"What about this evening?\"  \n"
        "The bot retains Bhopal + cycling automatically."
    )

    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.session_context = {"location": None, "activity": None, "time": None}
        st.session_state.pending_prompt = None
        st.rerun()

    st.caption("Starting a new session clears all remembered context.")

    st.markdown("---")

    # Decision pipeline (the former center flow, now clearly in sidebar)
    st.markdown("#### ⚙️ Decision Pipeline")
    st.caption("How your request is processed:")
    st.markdown("""
<div class="sidebar-pipeline">
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">1</span>
    <div class="pipeline-step-text"><b>Your Question</b><small>Extract intent &amp; context</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">2</span>
    <div class="pipeline-step-text"><b>Understand Context</b><small>Session memory &amp; time</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">3</span>
    <div class="pipeline-step-text"><b>Live Weather</b><small>Open-Meteo current &amp; forecast</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">4</span>
    <div class="pipeline-step-text"><b>Match Safety SOP</b><small>Deterministic policy evaluation</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">5</span>
    <div class="pipeline-step-text"><b>Policy Advisory</b><small>Traceable decision &amp; facts</small></div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.caption("The LLM handles language understanding. The deterministic SOP engine makes all safety decisions.")

# ---------------------------------------------------------------------------
# Main content area — header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="header-card">
  <p class="header-title">🌦️ Weather Advisory Support Bot</p>
  <p class="header-subtitle">Policy-based outdoor safety guidance using live weather data.</p>
  <p class="header-desc">Ask whether an outdoor activity is advisable based on current weather conditions and written safety policies. Every recommendation is traceable to a specific policy.</p>
</div>
""", unsafe_allow_html=True)

# What I Can Help With — Center card
st.markdown("""
<div class="help-card">
  <div class="help-header">
    <span class="help-title">💡 What I Can Help With</span>
    <span class="help-subtitle">Only activities with defined safety policies are supported.</span>
  </div>
  <div class="help-grid">
    <div class="help-chip">🚴 <span>Cycling</span></div>
    <div class="help-chip">🚶 <span>Walking</span></div>
    <div class="help-chip">🏃 <span>Running</span></div>
    <div class="help-chip">🚗 <span>Commuting</span></div>
    <div class="help-chip">🛵 <span>Scooter / Motorbike</span></div>
    <div class="help-chip">🧺 <span>Picnics &amp; outdoor recreation</span></div>
    <div class="help-chip">👨‍👩‍👧 <span>Children outdoors</span></div>
    <div class="help-chip">👴 <span>Elderly outdoor activities</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Example prompts — only show when chat is empty
# ---------------------------------------------------------------------------

EXAMPLES = [
    ("🚴", "Cycling", "Is it safe to cycle in Bhopal today?"),
    ("🚶", "Walking", "Is it okay to walk in Roorkee right now?"),
    ("👨‍👩‍👧", "Children", "Can I take my child to the park in Delhi this afternoon?"),
    ("🚗", "Travel", "Should I travel by two-wheeler in Jaipur today?"),
    ("🧺", "Picnic", "Is today a good day for a picnic in Chandigarh?"),
    ("🌅", "Follow-up", "What about this evening?"),
]

if not st.session_state.messages:
    st.markdown('<div class="examples-header">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (emoji, label, prompt) in enumerate(EXAMPLES):
        col = cols[i % 3]
        with col:
            if st.button(f"{emoji} {label}\n\"{prompt}\"", key=f"ex_{i}"):
                st.session_state.pending_prompt = prompt

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Handle pending prompt (from example button clicks)
# ---------------------------------------------------------------------------

prompt_to_run = None

if st.session_state.pending_prompt:
    prompt_to_run = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

if user_input := st.chat_input("Ask a weather-safety question…"):
    prompt_to_run = user_input

# ---------------------------------------------------------------------------
# Process prompt
# ---------------------------------------------------------------------------

if prompt_to_run:
    prompt = prompt_to_run

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run graph
    with st.chat_message("assistant"):
        with st.spinner("Checking live weather and safety policies…"):
            try:
                answer = run_graph(
                    user_message=prompt,
                    thread_id=st.session_state.thread_id,
                )
            except Exception as exc:
                answer = (
                    f"❌ **Unexpected Error**\n\n"
                    f"Something went wrong: `{exc}`\n\n"
                    f"Please ensure your `GEMINI_API_KEY` is set in your `.env` file."
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Update session context display from the answer ---
    # Parse location, activity, time from the answer text (best-effort)
    # This is display-only — actual context is managed by LangGraph MemorySaver
    _ctx = st.session_state.session_context

    # Extract location from answer text
    import re
    loc_match = re.search(r"(?:Weather Advisory\s*[—–-]+\s*|Current conditions in\s+)(.+?)(?:[\n\*]|\s*:\s*🌡️)", answer)
    if loc_match:
        raw_loc = loc_match.group(1).strip().rstrip("*")
        if raw_loc and raw_loc not in {"your location", ""}:
            _ctx["location"] = raw_loc

    # Extract activity from the prompt (simple keyword scan)
    _ACTIVITY_MAP = {
        "cycl": "Cycling", "bike": "Cycling", "bicycl": "Cycling",
        "walk": "Walking", "run": "Running", "jog": "Running",
        "picnic": "Picnic", "park": "Park visit",
        "scooter": "Scooter", "motorbike": "Motorbike",
        "travel": "Travel / commute", "commut": "Travel / commute",
        "hike": "Hiking", "trek": "Trekking",
    }
    prompt_lower = prompt.lower()
    for kw, label in _ACTIVITY_MAP.items():
        if kw in prompt_lower:
            _ctx["activity"] = label
            break

    # Extract time context from prompt
    _TIME_MAP = {
        "evening": "This evening", "morning": "This morning",
        "afternoon": "This afternoon", "night": "Tonight",
        "tomorrow": "Tomorrow", "today": "Today", "now": "Now",
    }
    for kw, label in _TIME_MAP.items():
        if kw in prompt_lower:
            _ctx["time"] = label
            break
    else:
        if not _ctx.get("time"):
            _ctx["time"] = "Current"

    st.session_state.session_context = _ctx
    st.rerun()
