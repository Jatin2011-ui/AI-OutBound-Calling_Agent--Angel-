"""LangGraph agent state definition.

``AgentState`` carries everything a node needs about the current call. List
fields use ``Annotated[..., operator.add]`` reducers so LangGraph *appends*
node outputs instead of overwriting them; ``metadata`` uses a shallow-merge
reducer so nodes don't clobber each other's keys.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp for conversation messages."""
    return datetime.now(timezone.utc)


class Message(TypedDict):
    role: Literal["agent", "caller"]
    content: str
    timestamp: datetime


def _merge_metadata(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class CallState:
    """Finite-state-machine state names for a call."""

    GREETING = "GREETING"
    QUERYING = "QUERYING"
    RESPONDING = "RESPONDING"
    ESCALATION = "ESCALATION"
    CLOSING = "CLOSING"


class CallOrchestratorState:
    """Per-call media-stream state machine for barge-in handling.

    States:
        IDLE        - no call activity
        GREETING    - agent delivering opening line
        LISTENING   - waiting for caller utterance
        THINKING    - LangGraph graph run in progress
        SPEAKING    - TTS audio streaming to Twilio
        INTERRUPTED - barge-in detected, cleaning up
        ESCALATION  - escalation webhook + DB in progress
        CLOSING     - goodbye / escalation / end-of-call in progress
    """

    IDLE = "IDLE"
    GREETING = "GREETING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ESCALATION = "ESCALATION"
    CLOSING = "CLOSING"


class AgentState(TypedDict, total=False):
    """Shared state threaded through every LangGraph node."""

    call_sid: str
    call_state: str
    conversation_history: Annotated[List[Message], operator.add]
    retrieved_memories: Annotated[List[str], operator.add]
    caller_intent: Optional[str]
    caller_sentiment: Optional[str]
    should_escalate: bool
    call_summary: Optional[str]
    turn_count: int
    metadata: Annotated[Dict[str, Any], _merge_metadata]


def new_agent_state(call_sid: str, metadata: Optional[Dict[str, Any]] = None) -> AgentState:
    """Create a freshly initialised state for a new call."""
    return AgentState(
        call_sid=call_sid,
        call_state=CallState.GREETING,
        conversation_history=[],
        retrieved_memories=[],
        caller_intent=None,
        caller_sentiment=None,
        should_escalate=False,
        call_summary=None,
        turn_count=0,
        metadata=dict(metadata or {}),
    )


def find_pending_caller_index(state: AgentState) -> Optional[int]:
    """Index of the first caller message not yet processed by the pipeline.

    Processing progress is tracked in ``state["metadata"]["processed_upto"]``
    (set by :mod:`app.agent.nodes.response_gen` after each reply). Returns
    ``None`` when there is nothing pending.
    """
    history = state.get("conversation_history", [])
    processed_upto = int(state.get("metadata", {}).get("processed_upto", -1))
    for index in range(processed_upto + 1, len(history)):
        if history[index]["role"] == "caller":
            return index
    return None


def apply_update(state: AgentState, update: Dict[str, Any]) -> None:
    """Merge a node's partial update into ``state`` (mirrors LangGraph reducers).

    Used when nodes are invoked directly (outside a compiled graph), e.g. by
    the live-call orchestrator in the media stream handler.
    """
    if not update:
        return
    if history := update.get("conversation_history"):
        state.setdefault("conversation_history", []).extend(history)
    if memories := update.get("retrieved_memories"):
        state.setdefault("retrieved_memories", []).extend(memories)
    if metadata := update.get("metadata"):
        merged = dict(state.get("metadata", {}))
        merged.update(metadata)
        state["metadata"] = merged
    for key, value in update.items():
        if key in ("conversation_history", "retrieved_memories", "metadata"):
            continue
        state[key] = value


__all__ = [
    "AgentState",
    "CallState",
    "Message",
    "utcnow",
    "new_agent_state",
    "find_pending_caller_index",
    "apply_update",
]
