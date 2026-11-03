"""Greeting node: delivers the agent's LLM-generated opening line exactly once per call."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.agent.state import AgentState, CallState, Message, utcnow
from app.api import dependencies

logger = logging.getLogger(__name__)


async def _llm_greeting(
    agent_name: str, company: str, context: Dict[str, Any]
) -> str:
    """Ask the LLM to produce a warm opening greeting based on call context."""
    customer = context.get("customer_name", "")
    purpose = str(context.get("purpose", "")).replace("_", " ").strip()
    context_lines = "\n".join(f"- {k}: {v}" for k, v in context.items()) or "- (none)"

    system_prompt = (
        f"You are {agent_name}, a friendly and professional outbound calling agent "
        f"for {company}.\n\n"
        "Your task: Generate a warm, natural-sounding opening greeting for an outbound call. "
        "Keep it to 1-2 sentences. Be friendly, introduce yourself and the company, "
        "and briefly mention the purpose if one is given in the context. "
        "Do NOT use words like 'certainly', 'sure', or 'absolutely'. "
        "Speak directly as if you are on the phone right now.\n\n"
        f"Call context:\n{context_lines}"
    )
    trigger: List[Message] = [
        Message(role="caller", content="[call connected]", timestamp=utcnow())
    ]
    responder = dependencies.get_response_generator()
    result = await responder(system_prompt=system_prompt, history=trigger)
    return (result or "").strip()


def _static_greeting(agent_name: str, company: str, context: Dict[str, Any]) -> str:
    """Fallback static greeting if the LLM is unavailable."""
    customer = context.get("customer_name", "")
    purpose = str(context.get("purpose", "")).replace("_", " ").strip()
    name_part = f", {customer}" if customer else ""
    purpose_part = f" regarding your {purpose}" if purpose else ""
    return (
        f"Hello{name_part}! This is {agent_name} calling from {company}"
        f"{purpose_part}. How can I help you today?"
    )


async def greeting_node(state: AgentState) -> Dict[str, Any]:
    """Set ``call_state = GREETING`` and append the LLM-generated opening message.

    Idempotent: if any agent message already exists in the history the node
    is a no-op, so re-entering the graph on later turns is safe.
    """
    if state.get("conversation_history"):
        return {}

    settings = dependencies.get_settings()
    context = state.get("metadata", {}).get("context", {})

    try:
        greeting_text = await _llm_greeting(
            settings.agent_name, settings.company_name, context
        )
    except Exception as exc:  # noqa: BLE001 - never crash the call on greeting failure
        logger.warning(
            "LLM greeting failed in greeting_node; using static fallback",
            extra={"call_sid": state.get("call_sid"), "error": str(exc)},
        )
        greeting_text = ""

    # Use the static fallback when the LLM returned empty (or raised above).
    if not greeting_text:
        greeting_text = _static_greeting(
            settings.agent_name, settings.company_name, context
        )

    logger.info(
        "Delivering LLM greeting",
        extra={"call_sid": state.get("call_sid"), "state": CallState.GREETING},
    )
    return {
        "call_state": CallState.GREETING,
        "conversation_history": [
            Message(role="agent", content=greeting_text, timestamp=utcnow())
        ],
    }


__all__ = ["greeting_node"]
