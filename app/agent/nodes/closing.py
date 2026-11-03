"""Closing node: wrap up the call.

Generates a summary of the conversation, persists the call record and full
transcript, appends a farewell message for the caller, and (best-effort)
upserts the resolved interaction into the memory store so future calls can
learn from it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.agent.state import AgentState, CallState, Message, utcnow
from app.api import dependencies
from app.memory.schemas import MemoryRecordCreate

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = (
    "You summarise phone calls. Given the conversation transcript, produce a "
    "concise 2-3 sentence summary covering: the call's purpose, the outcome, "
    "and any follow-up commitments made. Plain text only."
)


def _fallback_farewell(state: AgentState) -> str:
    customer = state.get("metadata", {}).get("context", {}).get("customer_name")
    name_part = f", {customer}" if customer else ""
    return f"Thank you for your time{name_part}. Have a great day. Goodbye!"


async def _generate_summary(state: AgentState) -> str:
    """Summarise the call via the responder, with a deterministic fallback."""
    history = list(state.get("conversation_history", []))
    if not history:
        return "Call connected but no conversation took place."
    try:
        responder = dependencies.get_response_generator()
        summary = await responder(system_prompt=SUMMARY_SYSTEM_PROMPT, history=history)
        return summary.strip() or _summarize_deterministically(state)
    except Exception as exc:  # noqa: BLE001 - never block call teardown
        logger.warning(
            "Summary generation failed; using fallback",
            extra={"call_sid": state.get("call_sid"), "error": str(exc)},
        )
        return _summarize_deterministically(state)


def _summarize_deterministically(state: AgentState) -> str:
    turns = len(state.get("conversation_history", []))
    intent = state.get("caller_intent") or "unknown"
    escalated = "escalated to a human agent" if state.get("should_escalate") else "handled by the AI agent"
    return f"Call with intent '{intent}' over {turns} turns; {escalated}."


async def closing_node(state: AgentState) -> Dict[str, Any]:
    """Finalise the call: summary, persistence, farewell message."""
    if state.get("metadata", {}).get("closing_completed"):
        return {}

    call_sid = state.get("call_sid", "unknown")
    logger.info("Closing call", extra={"call_sid": call_sid, "state": CallState.CLOSING})

    summary = await _generate_summary(state)

    # Farewell is composed here so it is part of the persisted transcript
    # (node partial updates are only merged into state after we return).
    farewell = Message(role="agent", content=_fallback_farewell(state), timestamp=utcnow())

    # Persist call record + transcript (best effort).
    try:
        persister = dependencies.get_call_persister()
        await persister.complete_call(
            call_sid=call_sid,
            summary=summary,
            history=list(state.get("conversation_history", [])) + [farewell],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to persist call closure",
            extra={"call_sid": call_sid, "error": str(exc)},
        )

    # Learn from resolved, non-escalated interactions.
    if not state.get("should_escalate"):
        try:
            store = dependencies.get_memory_store()
            await store.upsert(
                MemoryRecordCreate(
                    content=f"Resolved call {call_sid}: {summary}",
                    source="call_resolution",
                    metadata={
                        "call_sid": call_sid,
                        "outcome": "resolved",
                        "intent": state.get("caller_intent"),
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skipping memory write-back on close",
                extra={"call_sid": call_sid, "error": str(exc)},
            )

    return {
        "call_state": CallState.CLOSING,
        "call_summary": summary,
        "conversation_history": [farewell],
        "metadata": {"closing_completed": True, "ended_at": utcnow().isoformat()},
    }


__all__ = ["closing_node", "SUMMARY_SYSTEM_PROMPT"]
