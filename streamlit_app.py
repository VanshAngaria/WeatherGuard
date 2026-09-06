from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

import streamlit as st

# Page configuration - MUST be the first Streamlit command executed
try:
    st.set_page_config(
        page_title="Weather Advisory Support Bot",
        page_icon="🌦️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# Ensure repository root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# Detect Gemini API key from Streamlit Cloud Secrets if available
try:
    if hasattr(st, "secrets"):
        try:
            if bool(st.secrets) and "GEMINI_API_KEY" in st.secrets:
                key_val = st.secrets.get("GEMINI_API_KEY")
                if key_val:
                    os.environ["GEMINI_API_KEY"] = key_val
        except Exception:
            pass
except Exception:
    pass

from app.graph.graph import run_graph, run_graph_full

# Custom Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0f1117;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1080px;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

/* Unified Compact Guide Card */
.compact-guide-card {
    background: #121829;
    border: 1px solid #233354;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.9rem;
}

.compact-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    border-bottom: 1px solid #1c2740;
    padding-bottom: 0.55rem;
    margin-bottom: 0.6rem;
}

.compact-title-group {
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
}

.compact-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2edfa;
    letter-spacing: -0.2px;
}

.compact-subtitle {
    font-size: 0.77rem;
    color: #799ec2;
}

.compact-activity-pills {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.35rem;
}

.pills-label {
    font-size: 0.73rem;
    color: #5c7da8;
    font-weight: 600;
    margin-right: 2px;
}

.mini-pill {
    background: #172136;
    border: 1px solid #293a5c;
    border-radius: 4px;
    padding: 0.2rem 0.45rem;
    font-size: 0.72rem;
    color: #a8c4e0;
    font-weight: 500;
    white-space: nowrap;
}

.compact-steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.55rem;
}

@media (max-width: 900px) {
    .compact-steps-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 550px) {
    .compact-steps-grid {
        grid-template-columns: 1fr;
    }
}

.compact-step-item {
    background: #151d30;
    border: 1px solid #21314d;
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
}

.compact-step-badge {
    background: #1e2c4a;
    border: 1px solid #36507c;
    color: #82b1ff;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.66rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 3px;
}

.compact-step-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: #e0ecfb;
    margin-bottom: 2px;
}

.compact-step-desc {
    font-size: 0.71rem;
    color: #7292b7;
    margin-bottom: 4px;
    line-height: 1.25;
}

.compact-step-eg {
    font-size: 0.69rem;
    color: #64b5f6;
    background: #0d1424;
    padding: 2px 5px;
    border-radius: 3px;
    border: 1px solid #1a273f;
    line-height: 1.2;
    overflow-wrap: break-word;
}

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

.examples-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #5c7da8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.6rem;
}

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

section[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2d45 !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #7b9ec7;
}

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

.stChatMessage {
    border-radius: 10px !important;
}

hr {
    border: none !important;
    border-top: 1px solid #1e2d45 !important;
    margin: 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# State initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_context" not in st.session_state:
    st.session_state.session_context = {"location": None, "activity": None, "time": None}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="session-badge"><span class="session-dot"></span> System Online</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Session Context")
    st.caption("Maintains contextual continuity across follow-up queries.")

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

    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.session_context = {"location": None, "activity": None, "time": None}
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")

    st.markdown("#### ⚙️ Decision Pipeline")
    st.markdown("""
<div class="sidebar-pipeline">
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">1</span>
    <div class="pipeline-step-text"><b>Natural Language Parsing</b><small>Extract activity &amp; location intent</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">2</span>
    <div class="pipeline-step-text"><b>Session Context</b><small>LangGraph state &amp; time grounding</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">3</span>
    <div class="pipeline-step-text"><b>Live Weather Retrieval</b><small>Open-Meteo current &amp; forecast</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">4</span>
    <div class="pipeline-step-text"><b>SOP Rule Matching</b><small>Deterministic policy evaluation</small></div>
  </div>
  <div class="pipeline-step-connector">│</div>
  <div class="pipeline-step-item">
    <span class="pipeline-step-badge">5</span>
    <div class="pipeline-step-text"><b>Traceable Advisory</b><small>Structured facts &amp; safety guidance</small></div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.caption("LLM handles conversational parsing; the deterministic engine governs all safety logic.")

# Main content — Unified Compact Guide & Activities
st.markdown("""
<div class="compact-guide-card">
  <div class="compact-header-row">
    <div class="compact-title-group">
      <span class="compact-title">🌦️ WeatherGuard Advisory</span>
      <span class="compact-subtitle">Multi-Turn Conversational Safety Engine</span>
    </div>
    <div class="compact-activity-pills">
      <span class="pills-label">Supported:</span>
      <span class="mini-pill">🚴 Cycling</span>
      <span class="mini-pill">🚶 Walking</span>
      <span class="mini-pill">🏃 Running</span>
      <span class="mini-pill">🛵 Commuting</span>
      <span class="mini-pill">🧺 Recreation</span>
      <span class="mini-pill">👨‍👩‍👧 Children</span>
      <span class="mini-pill">👴 Elderly</span>
    </div>
  </div>
  <div class="compact-steps-grid">
    <div class="compact-step-item">
      <div class="compact-step-badge">Prompt 1</div>
      <div class="compact-step-name">Initial Query</div>
      <div class="compact-step-desc">Activity + place or direct city</div>
      <div class="compact-step-eg">💬 <i>"Is it safe to ride in Mumbai?"</i></div>
    </div>
    <div class="compact-step-item">
      <div class="compact-step-badge">Prompt 2</div>
      <div class="compact-step-name">Time Follow-up</div>
      <div class="compact-step-desc">Shift time; retains city &amp; activity</div>
      <div class="compact-step-eg">💬 <i>"What about this evening?"</i></div>
    </div>
    <div class="compact-step-item">
      <div class="compact-step-badge">Prompt 3</div>
      <div class="compact-step-name">Location Switch</div>
      <div class="compact-step-desc">Switch city; keeps activity &amp; time</div>
      <div class="compact-step-eg">💬 <i>"What about in Delhi?"</i></div>
    </div>
    <div class="compact-step-item">
      <div class="compact-step-badge">Prompt 4</div>
      <div class="compact-step-name">Activity Switch</div>
      <div class="compact-step-desc">Switch activity for same location</div>
      <div class="compact-step-eg">💬 <i>"What about walking?"</i></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Example queries
EXAMPLES = [
    ("🛵", "Ride / Scooter", "Is it safe to ride in Mumbai?"),
    ("🌦️", "Weather Guidance", "Weather guidance of Zirakpur"),
    ("📍", "Direct Location", "Zirakpur"),
    ("🌅", "Time Follow-up", "What about this evening?"),
    ("🚶", "Activity Follow-up", "What about walking?"),
    ("📍", "Location Follow-up", "What about in Delhi?"),
]

if not st.session_state.messages:
    st.markdown('<div class="examples-header">💡 Try Quick Inquiries</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (emoji, label, prompt) in enumerate(EXAMPLES):
        col = cols[i % 3]
        with col:
            if st.button(f"{emoji} {label}\n\"{prompt}\"", key=f"ex_{i}"):
                st.session_state.pending_prompt = prompt

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Prompt handling
prompt_to_run = None
if st.session_state.pending_prompt:
    prompt_to_run = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input := st.chat_input("Ask about weather safety for an activity…"):
    prompt_to_run = user_input

if prompt_to_run:
    prompt = prompt_to_run

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing live weather and safety policies…"):
            try:
                answer, res_state = run_graph_full(
                    user_message=prompt,
                    thread_id=st.session_state.thread_id,
                )
            except Exception as exc:
                answer = (
                    f"❌ **Service Notice**\n\n"
                    f"Could not complete evaluation: `{exc}`\n\n"
                    f"Please verify that `GEMINI_API_KEY` is configured in Streamlit secrets or `.env`."
                )
                res_state = {}
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Update session context for display directly from graph state
    _ctx = st.session_state.session_context

    loc_val = res_state.get("resolved_location") or res_state.get("location_text")
    if loc_val and loc_val not in {"your location", ""}:
        _ctx["location"] = loc_val

    act_val = res_state.get("activity")
    if act_val and act_val not in {"general", "this activity", ""}:
        _ctx["activity"] = act_val.title()

    time_val = res_state.get("requested_time")
    if time_val:
        _ctx["time"] = time_val.title()

    st.session_state.session_context = _ctx
    st.rerun()
