"""Response generation node.

Builds a context-enriched prompt (system prompt + retrieved memory chunks +
conversation history + caller intent) and asks the configured responder for
the agent's next line. The responder is injectable so the same node serves
offline/batch runs (chat-completions backed) and tests (scripted fakes);
during live calls the generated text is synthesised via Deepgram TTS.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.agent.state import AgentState, CallState, Message, find_pending_caller_index, utcnow
from app.api import dependencies

logger = logging.getLogger(__name__)

# Fallback lines cycled on consecutive LLM failures — never repeat verbatim.
FALLBACK_REPLIES = (
    "I'm sorry, I didn't quite catch that. Could you say it again?",
    "Apologies, the line seems a bit unclear. Could you repeat that once more?",
    "I'm having persistent trouble processing this call. Let me have a team "
    "member follow up with you directly.",
)

MAX_CONSECUTIVE_LLM_FAILURES = 3  # after this, the call closes


def build_response_system_prompt(state: AgentState) -> str:
    """Assemble the system prompt with retrieved memory context inline."""
    settings = dependencies.get_settings()
    context = state.get("metadata", {}).get("context", {})
    base = settings.load_system_prompt(context)

    memories: List[str] = state.get("retrieved_memories", [])
    if memories:
        memory_block = "\n".join(f"- {chunk}" for chunk in memories)
        base += (
            "\n\nRelevant knowledge (use only if it helps answer the caller):\n"
            f"{memory_block}"
        )

    intent = state.get("caller_intent")
    if intent:
        base += f"\n\nDetected caller intent: {intent}."

    return base


async def response_gen_node(state: AgentState) -> Dict[str, Any]:
    """Generate the agent's reply to the pending caller message."""
    call_sid = state.get("call_sid", "unknown")

    pending_index = find_pending_caller_index(state)
    if pending_index is None:
        # Nothing to respond to (already answered, or no caller turn yet).
        return {}

    responder = dependencies.get_response_generator()
    system_prompt = build_response_system_prompt(state)
    history: List[Message] = list(state.get("conversation_history", []))

    llm_failures = int(state.get("metadata", {}).get("llm_failures", 0))

    try:
        reply_text = await responder(system_prompt=system_prompt, history=history)
        llm_failures = 0  # reset the counter on success
    except Exception as exc:  # noqa: BLE001 - keep the call alive on failure
        llm_failures += 1
        logger.error(
            "Response generation failed; using fallback",
            extra={
                "call_sid": call_sid,
                "error": str(exc),
                "consecutive_llm_failures": llm_failures,
            },
        )
        reply_text = FALLBACK_REPLIES[min(llm_failures - 1, len(FALLBACK_REPLIES) - 1)]

    reply_text = (reply_text or "").strip() or FALLBACK_REPLIES[0]

    turn_count = int(state.get("turn_count", 0)) + 1
    logger.info(
        "Generated response",
        extra={
            "call_sid": call_sid,
            "state": CallState.RESPONDING,
            "turn": turn_count,
            "llm_failures": llm_failures,
            "fallback": bool(llm_failures),
        },
    )
    return {
        "call_state": CallState.RESPONDING,
        "conversation_history": [Message(role="agent", content=reply_text, timestamp=utcnow())],
        "turn_count": turn_count,
        "metadata": {"processed_upto": pending_index, "llm_failures": llm_failures},
    }


__all__ = ["response_gen_node", "build_response_system_prompt"]
