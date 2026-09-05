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
from app.graph.nodes.scope_response import scope_response_node
from app.graph.state import BotState

logger = logging.getLogger(__name__)


def route_after_intent(
    state: BotState,
) -> Literal["scope_response", "clarification_response", "resolve_location", "error_response"]:
    """Route after intent parsing based on scope, clarification, or errors."""
    if state.get("error_type"):
        return "error_response"
    if state.get("scope_type") == "irrelevant":
        return "scope_response"
    if state.get("needs_clarification"):
        return "clarification_response"
    return "resolve_location"


def route_after_location(
    state: BotState,
) -> Literal["fetch_weather", "error_response"]:
    """Route to error_response if geocoding failed."""
    if state.get("error_type"):
        return "error_response"
    return "fetch_weather"


def route_after_weather(
    state: BotState,
) -> Literal["extract_facts", "error_response"]:
    """Route to error_response if weather data fetching failed."""
    if state.get("error_type"):
        return "error_response"
    return "extract_facts"


def route_after_extract(
    state: BotState,
) -> Literal["match_sops", "error_response"]:
    """Route to error_response if fact validation failed."""
    if state.get("error_type"):
        return "error_response"
    return "match_sops"


def route_after_match(
    state: BotState,
) -> Literal["resolve_policy", "no_match_response"]:
    """Route to no_match_response if no policy conditions were satisfied."""
    matches = state.get("sop_matches")
    if not matches:
        return "no_match_response"
    return "resolve_policy"


def build_graph() -> StateGraph:
    """Build and compile the conversational LangGraph state machine with memory persistence."""
    builder = StateGraph(BotState)

    builder.add_node("parse_intent", parse_intent_node)
    builder.add_node("scope_response", scope_response_node)
    builder.add_node("clarification_response", clarification_response_node)
    builder.add_node("resolve_location", resolve_location_node)
    builder.add_node("fetch_weather", fetch_weather_node)
    builder.add_node("extract_facts", extract_facts_node)
    builder.add_node("match_sops", match_sops_node)
    builder.add_node("resolve_policy", resolve_policy_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("error_response", error_response_node)
    builder.add_node("no_match_response", no_match_response_node)

    # Core flow
    builder.add_edge(START, "parse_intent")
    builder.add_edge("resolve_policy", "generate_response")
    builder.add_edge("generate_response", END)
    builder.add_edge("scope_response", END)
    builder.add_edge("clarification_response", END)
    builder.add_edge("error_response", END)
    builder.add_edge("no_match_response", END)

    # Conditional branching
    builder.add_conditional_edges(
        "parse_intent",
        route_after_intent,
        {
            "scope_response": "scope_response",
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

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


_graph = None


def get_graph():
    """Return singleton instance of the compiled LangGraph workflow."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_graph(user_message: str, thread_id: str = "default") -> str:
    """Run a single conversational turn through the state graph."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    input_state: BotState = {
        "user_message": user_message,
        "thread_id": thread_id,
    }

    result = graph.invoke(input_state, config=config)
    return result.get("final_answer", "I encountered an unexpected error.")
