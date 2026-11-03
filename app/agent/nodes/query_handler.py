"""Query handler node: intent / sentiment extraction and escalation triggers.

The default classifier is a transparent rule-based implementation so the
pipeline is deterministic and testable. It can be swapped for an LLM-backed
extractor via ``dependencies.set_override("intent_classifier", ...)`` — the
callable must have the signature ``(text: str) -> tuple[str, str]`` returning
``(intent, sentiment)``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from app.agent.state import AgentState, find_pending_caller_index
from app.api import dependencies

logger = logging.getLogger(__name__)

# Intents that signal the caller wants to end the conversation.
END_CALL_INTENTS = frozenset({"end_call"})

_END_PHRASES = (
    "goodbye",
    "bye",
    "that's all",
    "thats all",
    "that is all",
    "nothing else",
    "no thanks",
    "no thank you",
    "i'm good",
    "im good",
    "gotta go",
)
_NEGATIVE_MARKERS = ("angry", "upset", "frustrat", "terrible", "awful", "ridiculous", "unhappy")
_POSITIVE_MARKERS = ("thank", "great", "perfect", "wonderful", "awesome", "appreciate")


def heuristic_classify(text: str) -> Tuple[str, str]:
    """Deterministically classify intent and sentiment from caller text."""
    lowered = text.lower()

    if any(phrase in lowered for phrase in _END_PHRASES):
        intent = "end_call"
    elif "reschedul" in lowered or "move it" in lowered or "change the" in lowered:
        intent = "reschedule"
    elif "cancel" in lowered:
        intent = "cancellation"
    elif "appointment" in lowered or "booking" in lowered or "schedule" in lowered:
        intent = "appointment_inquiry"
    else:
        intent = "general_inquiry"

    if any(marker in lowered for marker in _NEGATIVE_MARKERS):
        sentiment = "negative"
    elif any(marker in lowered for marker in _POSITIVE_MARKERS):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return intent, sentiment


async def query_handler_node(state: AgentState) -> Dict[str, Any]:
    """Extract intent/sentiment for the pending caller message.

    Also sets ``should_escalate`` if any configured escalation keyword appears
    in the caller's utterance.
    """
    pending_index = find_pending_caller_index(state)
    if pending_index is None:
        return {}

    utterance = state["conversation_history"][pending_index]["content"]
    call_sid = state.get("call_sid", "unknown")

    try:
        classifier = dependencies.get_intent_classifier()
        intent, sentiment = classifier(utterance)
    except Exception as exc:  # noqa: BLE001 - never block a call on analysis
        logger.warning(
            "Intent classification failed; using fallback",
            extra={"call_sid": call_sid, "error": str(exc)},
        )
        intent, sentiment = "general_inquiry", "neutral"

    settings = dependencies.get_settings()
    lowered = utterance.lower()
    should_escalate = bool(state.get("should_escalate")) or any(
        keyword in lowered for keyword in settings.escalation_keyword_list
    )

    logger.info(
        "Query analysed",
        extra={
            "call_sid": call_sid,
            "intent": intent,
            "sentiment": sentiment,
            "escalate": should_escalate,
        },
    )
    return {
        "caller_intent": intent,
        "caller_sentiment": sentiment,
        "should_escalate": should_escalate,
    }


__all__ = ["query_handler_node", "heuristic_classify", "END_CALL_INTENTS"]
