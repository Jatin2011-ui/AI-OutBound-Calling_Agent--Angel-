"""Conditional routing logic for the LangGraph agent.

``route_after_response`` is the single conditional edge out of the
``response_gen`` node. It must return one of ``"continue"`` (loop back to
``memory_retrieval`` for the next caller turn), ``"escalate"``, or
``"close"``.
"""

from __future__ import annotations

import logging
from typing import Literal

from app.agent.state import AgentState, find_pending_caller_index

logger = logging.getLogger(__name__)

RouteDecision = Literal["continue", "escalate", "close"]

# Intents that mean the caller is done (kept in sync with query_handler).
CLOSING_INTENTS = frozenset({"end_call", "goodbye", "closing"})

# Safety net if settings aren't reachable (e.g. exotic test contexts).
_FALLBACK_MAX_TURNS = 20


def _max_turns(state: AgentState) -> int:
    try:
        from app.api import dependencies

        return dependencies.get_settings().max_turns_before_close
    except Exception:  # noqa: BLE001 - routing must never raise
        return int(state.get("metadata", {}).get("max_turns", _FALLBACK_MAX_TURNS))


def route_after_response(state: AgentState) -> RouteDecision:
    """Decide where the graph goes after ``response_gen``.

    - ``"escalate"`` when the caller triggered an escalation keyword or the
      intent classifier flagged the call.
    - ``"close"`` when the caller signalled the end of the conversation, the
      turn budget is exhausted, the LLM has failed too many turns in a row,
      or no caller message remains to process.
    - ``"continue"`` otherwise (more caller messages are pending handling).
    """
    from app.agent.nodes.response_gen import MAX_CONSECUTIVE_LLM_FAILURES

    call_sid = state.get("call_sid", "unknown")
    turn_count = state.get("turn_count", 0)

    if state.get("should_escalate"):
        logger.info("Routing → escalate", extra={"call_sid": call_sid, "turn": turn_count})
        return "escalate"

    intent = state.get("caller_intent") or ""
    if intent in CLOSING_INTENTS:
        logger.info(
            "Routing → close (end intent)",
            extra={"call_sid": call_sid, "turn": turn_count, "intent": intent},
        )
        return "close"

    llm_failures = int(state.get("metadata", {}).get("llm_failures", 0))
    if llm_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
        logger.warning(
            "Routing → close (LLM unavailable for %d consecutive turns)",
            llm_failures,
            extra={"call_sid": call_sid, "turn": turn_count},
        )
        return "close"

    if turn_count >= _max_turns(state):
        logger.info(
            "Routing → close (max turns)",
            extra={"call_sid": call_sid, "turn": turn_count},
        )
        return "close"

    if find_pending_caller_index(state) is not None:
        logger.info("Routing → continue", extra={"call_sid": call_sid, "turn": turn_count})
        return "continue"

    logger.info(
        "Routing → close (no pending caller messages)",
        extra={"call_sid": call_sid, "turn": turn_count},
    )
    return "close"


__all__ = ["route_after_response", "RouteDecision", "CLOSING_INTENTS"]
