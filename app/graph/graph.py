"""
LangGraph graph definition.
Wires all nodes into a StateGraph with real conditional branches.

Graph flow:
  START
    ↓
  parse_intent
    ├── needs_clarification ─────────────────→ clarification_response → END
    ↓
  resolve_location ──── failure ──→ error_response → END
    ↓
  fetch_weather ──────── failure ──→ error_response → END
    ↓
  extract_facts ─────── failure ──→ error_response → END
    ↓
  match_sops
    ├── no match ──────────────────────────→ no_match_response → END
    ↓
  resolve_policy
    ↓
  generate_response    ← LLM composes language from structured policy decision
    ↓
  END

Branching summary:
  parse_intent    → clarification_response (ambiguous) | resolve_location (normal)
  resolve_location → error_response (failure) | fetch_weather (success)
  fetch_weather    → error_response (failure) | extract_facts (success)
  extract_facts    → error_response (failure) | match_sops (success)
  match_sops       → no_match_response (empty) | resolve_policy (matches found)
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.clarification_response import clarification_response_node
from app.graph.nodes.error_response import error_response_node
from app.graph.nodes.extract_facts import extract_facts_node
from app.graph.nodes.fetch_weather import fetch_weather_node
from app.graph.nodes.generate_response import generate_response_node
from app.graph.nodes.match_sops import match_sops_node
from app.graph.nodes.no_match_response import no_match_response_node
from app.graph.nodes.parse_intent import parse_intent_node
from app.graph.nodes.resolve_location import resolve_location_node
from app.graph.nodes.resolve_policy import resolve_policy_node
from app.graph.state import BotState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_intent(
    state: BotState,
) -> Literal["clarification_response", "resolve_location", "error_response"]:
    """Route to clarification if ambiguous; error if LLM failed; normal otherwise."""
    if state.get("error_type"):
        logger.debug("Routing: intent failure → error_response")
        return "error_response"
    if state.get("needs_clarification"):
        logger.debug("Routing: ambiguous intent → clarification_response")
        return "clarification_response"
    return "resolve_location"


def route_after_location(
    state: BotState,
) -> Literal["fetch_weather", "error_response"]:
    """Route to error_response if location resolution failed."""
    if state.get("error_type"):
        logger.debug("Routing: location failure → error_response")
        return "error_response"
    return "fetch_weather"


def route_after_weather(
    state: BotState,
) -> Literal["extract_facts", "error_response"]:
    """Route to error_response if weather fetch failed."""
    if state.get("error_type"):
        logger.debug("Routing: weather failure → error_response")
        return "error_response"
    return "extract_facts"


def route_after_extract(
    state: BotState,
) -> Literal["match_sops", "error_response"]:
    """Route to error_response if fact extraction failed."""
    if state.get("error_type"):
        logger.debug("Routing: extract failure → error_response")
        return "error_response"
    return "match_sops"


def route_after_match(
    state: BotState,
) -> Literal["resolve_policy", "no_match_response"]:
    """Route to no_match_response if no SOPs matched."""
    matches = state.get("sop_matches")
    if not matches:
        logger.debug("Routing: no SOP matches → no_match_response")
        return "no_match_response"
    return "resolve_policy"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.
    Uses MemorySaver for in-memory checkpointing (session memory).
    """
    builder = StateGraph(BotState)

    # --- Add nodes ---
    builder.add_node("parse_intent", parse_intent_node)
    builder.add_node("clarification_response", clarification_response_node)
    builder.add_node("resolve_location", resolve_location_node)
    builder.add_node("fetch_weather", fetch_weather_node)
    builder.add_node("extract_facts", extract_facts_node)
    builder.add_node("match_sops", match_sops_node)
    builder.add_node("resolve_policy", resolve_policy_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("error_response", error_response_node)
    builder.add_node("no_match_response", no_match_response_node)

    # --- Fixed edges ---
    builder.add_edge(START, "parse_intent")
    builder.add_edge("resolve_policy", "generate_response")
    builder.add_edge("generate_response", END)
    builder.add_edge("clarification_response", END)
    builder.add_edge("error_response", END)
    builder.add_edge("no_match_response", END)

    # --- Conditional edges (real branching) ---
    builder.add_conditional_edges(
        "parse_intent",
        route_after_intent,
        {
            "clarification_response": "clarification_response",
            "resolve_location": "resolve_location",
            "error_response": "error_response",
        },
    )
    builder.add_conditional_edges(
        "resolve_location",
        route_after_location,
        {"fetch_weather": "fetch_weather", "error_response": "error_response"},
    )
    builder.add_conditional_edges(
        "fetch_weather",
        route_after_weather,
        {"extract_facts": "extract_facts", "error_response": "error_response"},
    )
    builder.add_conditional_edges(
        "extract_facts",
        route_after_extract,
        {"match_sops": "match_sops", "error_response": "error_response"},
    )
    builder.add_conditional_edges(
        "match_sops",
        route_after_match,
        {"resolve_policy": "resolve_policy", "no_match_response": "no_match_response"},
    )

    # Compile with in-memory checkpointer for session memory
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    logger.info("LangGraph graph compiled successfully.")
    return graph


# Module-level singleton
_graph = None


def get_graph():
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_graph(user_message: str, thread_id: str = "default") -> str:
    """
    Run one conversation turn through the graph.

    Args:
        user_message: The user's input text.
        thread_id:    Session/thread identifier for memory continuity.

    Returns:
        The final answer string.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    input_state: BotState = {
        "user_message": user_message,
        "thread_id": thread_id,
    }

    logger.info("Running graph: thread=%s message='%s'", thread_id, user_message)

    result = graph.invoke(input_state, config=config)
    answer = result.get("final_answer", "I encountered an unexpected error.")
    return answer
