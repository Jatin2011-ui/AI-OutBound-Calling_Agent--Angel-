"""Memory retrieval node: pgvector similarity search for the caller's turn.

Embeds the caller's latest unanswered utterance and pulls the top-k most
relevant knowledge chunks from the long-term memory store. Failures degrade
to zero context — the conversation must never die on a memory outage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.agent.state import AgentState, CallState, find_pending_caller_index
from app.api import dependencies

logger = logging.getLogger(__name__)


async def memory_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve relevant memory chunks for the pending caller message."""
    pending_index = find_pending_caller_index(state)
    if pending_index is None:
        return {}

    history = state.get("conversation_history", [])
    utterance = history[pending_index]["content"]
    call_sid = state.get("call_sid", "unknown")

    try:
        store = dependencies.get_memory_store()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning(
            "Memory store unavailable; continuing without context",
            extra={"call_sid": call_sid, "error": str(exc)},
        )
        return {"call_state": CallState.QUERYING}

    try:
        results = await store.search(utterance)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.error(
            "Memory retrieval failed",
            extra={"call_sid": call_sid, "error": str(exc)},
        )
        return {"call_state": CallState.QUERYING}

    chunks = [r.content for r in results]
    logger.info(
        "Retrieved memory chunks",
        extra={"call_sid": call_sid, "state": CallState.QUERYING, "hits": len(chunks)},
    )
    return {
        "call_state": CallState.QUERYING,
        "retrieved_memories": chunks,
        "metadata": {"memory_retrieved_for_index": pending_index},
    }


__all__ = ["memory_retrieval_node"]
