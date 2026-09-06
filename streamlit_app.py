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
            if "GEMINI_API_KEY" in st.secrets:
                os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
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
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 880px;
}

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

# Main content
st.markdown("""
<div class="header-card">
  <p class="header-title">🌦️ Weather Advisory Support Bot</p>
  <p class="header-subtitle">Policy-governed outdoor safety guidance grounded in verified live weather data.</p>
  <p class="header-desc">Ask whether outdoor activities are safe under current atmospheric conditions. Every safety recommendation is strictly traceable to a verified Standard Operating Procedure (SOP).</p>
</div>
""", unsafe_allow_html=True)

# Centered Supported Activities Card
st.markdown("""
<div class="help-card">
  <div class="help-header">
    <span class="help-title">💡 Supported Activities</span>
    <span class="help-subtitle">Only activities with defined safety policies are evaluated.</span>
  </div>
  <div class="help-grid">
    <div class="help-chip">🚴 <span>Cycling</span></div>
    <div class="help-chip">🚶 <span>Walking</span></div>
    <div class="help-chip">🏃 <span>Running</span></div>
    <div class="help-chip">🚗 <span>Commuting</span></div>
    <div class="help-chip">🛵 <span>Scooter / Motorbike</span></div>
    <div class="help-chip">🧺 <span>Outdoor Recreation</span></div>
    <div class="help-chip">👨‍👩‍👧 <span>Children Outdoors</span></div>
    <div class="help-chip">👴 <span>Elderly Activities</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Example queries
EXAMPLES = [
    ("🚴", "Cycling", "Is it safe to cycle in Bhopal today?"),
    ("🚶", "Walking", "Is it okay to walk in Roorkee right now?"),
    ("👨‍👩‍👧", "Children", "Can I take my child to the park in Delhi this afternoon?"),
    ("🚗", "Travel", "Should I travel by two-wheeler in Jaipur today?"),
    ("🧺", "Picnic", "Is today a good day for a picnic in Chandigarh?"),
    ("🌅", "Follow-up", "What about this evening?"),
]

if not st.session_state.messages:
    st.markdown('<div class="examples-header">Example Inquiries</div>', unsafe_allow_html=True)
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
