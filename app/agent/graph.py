"""LangGraph StateGraph assembly for the outbound calling agent.

Topology (per the project README)::

    START → greeting → memory_retrieval → query_handler → response_gen
                                                            │
                              ┌─────────────────────────────┼──────────────┐
                          "continue"                    "escalate"      "close"
                              │                             │              │
                    memory_retrieval ◀── (next caller   escalation        │
                              │          turn)             │              │
                              └─ ...                       ▼              ▼
                                                        closing ◀─────────┘
                                                           │
                                                          END

The graph is compiled without a checkpointer (stateless per invocation). A
single ``ainvoke`` processes every caller message present in
``conversation_history`` in order, looping through ``memory_retrieval`` via
the ``"continue"`` route until all pending messages have been answered.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.edges.conditions import route_after_response
from app.agent.nodes import (
    closing_node,
    escalation_node,
    greeting_node,
    memory_retrieval_node,
    query_handler_node,
    response_gen_node,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_agent_graph() -> CompiledStateGraph:
    """Assemble and compile the agent graph (no checkpointer)."""
    graph = StateGraph(AgentState)

    # Register nodes.
    graph.add_node("greeting", greeting_node)
    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("query_handler", query_handler_node)
    graph.add_node("response_gen", response_gen_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("closing", closing_node)

    # Define edges.
    graph.set_entry_point("greeting")
    graph.add_edge("greeting", "memory_retrieval")
    graph.add_edge("memory_retrieval", "query_handler")
    graph.add_edge("query_handler", "response_gen")
    graph.add_conditional_edges(
        "response_gen",
        route_after_response,
        {
            "continue": "memory_retrieval",
            "escalate": "escalation",
            "close": "closing",
        },
    )
    graph.add_edge("escalation", "closing")
    graph.add_edge("closing", END)

    agent = graph.compile(checkpointer=None)
    logger.debug("Agent graph compiled")
    return agent


__all__ = ["build_agent_graph"]
