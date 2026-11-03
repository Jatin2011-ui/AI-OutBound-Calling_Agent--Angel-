"""Escalation node: flag the call for a human and notify the on-call channel."""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from app.agent.state import AgentState, CallState, Message, utcnow
from app.api import dependencies

logger = logging.getLogger(__name__)

HOLD_MESSAGE = (
    "Of course — let me connect you with a member of our team who can help "
    "you directly. Please hold for a moment."
)


async def _notify_human_agent(webhook_url: str, call_sid: str, state: AgentState) -> None:
    """POST a lightweight escalation notification (Slack-style) best-effort."""
    payload = {
        "text": (
            f"🚨 Escalation requested on call {call_sid}. "
            f"Intent: {state.get('caller_intent')}, "
            f"sentiment: {state.get('caller_sentiment')}."
        )
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(webhook_url, json=payload)
    except httpx.HTTPError as exc:
        logger.warning(
            "Escalation webhook notification failed",
            extra={"call_sid": call_sid, "error": str(exc)},
        )


async def escalation_node(state: AgentState) -> Dict[str, Any]:
    """Mark the call escalated, queue a hold message, and notify humans."""
    call_sid = state.get("call_sid", "unknown")
    logger.info("Escalating call", extra={"call_sid": call_sid, "state": CallState.ESCALATION})

    # Best-effort persistence of the escalated status.
    try:
        persister = dependencies.get_call_persister()
        await persister.mark_escalated(call_sid)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not persist escalation status",
            extra={"call_sid": call_sid, "error": str(exc)},
        )

    # Optional human notification.
    settings = dependencies.get_settings()
    if settings.escalation_webhook_url:
        await _notify_human_agent(settings.escalation_webhook_url, call_sid, state)

    return {
        "call_state": CallState.ESCALATION,
        "conversation_history": [
            Message(role="agent", content=HOLD_MESSAGE, timestamp=utcnow())
        ],
        "metadata": {"escalated_at": utcnow().isoformat()},
    }


__all__ = ["escalation_node", "HOLD_MESSAGE"]
