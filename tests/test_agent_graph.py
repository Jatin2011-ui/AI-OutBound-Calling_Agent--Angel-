"""Tests for the LangGraph agent: nodes, conditional routing, compiled graph."""

from __future__ import annotations

import uuid

from app.agent.edges.conditions import route_after_response
from app.agent.graph import build_agent_graph
from app.agent.nodes import (
    closing_node,
    escalation_node,
    greeting_node,
    memory_retrieval_node,
    query_handler_node,
    response_gen_node,
)
from app.agent.state import CallState
from app.memory.schemas import MemoryRecordCreate

from app.memory.schemas import MemorySearchResult

from .conftest import agent_message, caller_message


# ── Greeting node ────────────────────────────────────────────────────────────


async def test_greeting_delivers_opening_once(base_state):
    first = await greeting_node(base_state)
    assert first["call_state"] == CallState.GREETING
    assert len(first["conversation_history"]) == 1
    message = first["conversation_history"][0]
    assert message["role"] == "agent"
    assert "Priya Sharma" in message["content"]

    # Second invocation is a no-op (idempotent across turns).
    base_state["conversation_history"].extend(first["conversation_history"])
    assert await greeting_node(base_state) == {}


# ── Memory retrieval node ────────────────────────────────────────────────────


async def test_memory_retrieval_collects_chunks(base_state, fake_memory_store):
    fake_memory_store.results = [
        MemorySearchResult(
            id=uuid.uuid4(),
            content="Free rescheduling up to 24 hours before.",
            source="policy_doc",
            metadata={},
            distance=0.12,
        )
    ]
    base_state["conversation_history"].append(caller_message("Can I reschedule?"))

    update = await memory_retrieval_node(base_state)

    assert update["call_state"] == CallState.QUERYING
    assert update["retrieved_memories"] == ["Free rescheduling up to 24 hours before."]
    assert fake_memory_store.searches == ["Can I reschedule?"]


async def test_memory_retrieval_degrades_without_store(base_state):
    # No memory_store override registered and no live pool → graceful no-op.
    base_state["conversation_history"].append(caller_message("Hello?"))
    update = await memory_retrieval_node(base_state)
    assert update.get("retrieved_memories", []) == []


# ── Query handler node ───────────────────────────────────────────────────────


async def test_query_handler_extracts_intent_and_sentiment(base_state):
    base_state["conversation_history"].append(
        caller_message("Actually, I need to reschedule. Thanks!")
    )
    update = await query_handler_node(base_state)
    assert update["caller_intent"] == "reschedule"
    assert update["caller_sentiment"] == "positive"
    assert update["should_escalate"] is False


async def test_query_handler_flags_escalation_keyword(base_state):
    base_state["conversation_history"].append(
        caller_message("This is ridiculous. Put a manager on the phone.")
    )
    update = await query_handler_node(base_state)
    assert update["should_escalate"] is True
    assert update["caller_sentiment"] == "negative"


# ── Response generation node ────────────────────────────────────────────────


async def test_response_gen_answers_pending_message(base_state, fake_responder):
    fake_responder.replies = ["Your appointment is on October 1st at 10 AM."]
    base_state["conversation_history"].append(caller_message("What is this about?"))
    base_state["retrieved_memories"] = ["Policy chunk"]

    update = await response_gen_node(base_state)

    assert update["turn_count"] == 1
    assert update["call_state"] == CallState.RESPONDING
    assert update["conversation_history"][0]["content"] == "Your appointment is on October 1st at 10 AM."
    # Memory chunks and intent are injected into the responder's system prompt.
    assert "Policy chunk" in fake_responder.calls[0]["system_prompt"]


async def test_response_gen_noop_without_pending_caller(base_state, fake_responder):
    base_state["conversation_history"].append(agent_message("Hi there!"))
    assert await response_gen_node(base_state) == {}
    assert fake_responder.calls == []


class FailingResponder:
    """Responder that always raises (simulates a dead LLM provider)."""

    async def __call__(self, *, system_prompt, history):
        raise RuntimeError("groq: 401 invalid api key")


async def test_response_gen_fallback_never_repeats_and_counts_failures(base_state):
    from app.api import dependencies
    from app.agent.nodes.response_gen import FALLBACK_REPLIES

    dependencies.set_override("response_generator", FailingResponder())

    replies = []
    for _ in range(3):
        base_state["conversation_history"].append(caller_message("hello?"))
        update = await response_gen_node(base_state)
        base_state["metadata"].update(update["metadata"])
        base_state["turn_count"] = update["turn_count"]
        base_state["conversation_history"].extend(update["conversation_history"])
        replies.append(update["conversation_history"][0]["content"])

    assert replies == list(FALLBACK_REPLIES)
    assert len(set(replies)) == 3, "fallback lines must not repeat"
    assert base_state["metadata"]["llm_failures"] == 3


async def test_response_gen_failure_counter_resets_on_success(base_state, fake_responder):
    from app.api import dependencies

    dependencies.set_override("response_generator", FailingResponder())
    base_state["conversation_history"].append(caller_message("hi"))
    first = await response_gen_node(base_state)
    assert first["metadata"]["llm_failures"] == 1

    # Provider recovers — swap back to the working responder
    dependencies.set_override("response_generator", fake_responder)
    fake_responder.replies = ["Back on track!"]
    base_state["conversation_history"].extend(first["conversation_history"])
    base_state["metadata"].update(first["metadata"])
    base_state["conversation_history"].append(caller_message("are you there?"))
    second = await response_gen_node(base_state)
    assert second["metadata"]["llm_failures"] == 0
    assert second["conversation_history"][0]["content"] == "Back on track!"


# ── Closing / escalation nodes ───────────────────────────────────────────────


async def test_closing_generates_summary_and_farewell(base_state, fake_responder, fake_persister):
    fake_responder.replies = ["Caller confirmed appointment for Oct 1 at 10 AM."]
    base_state["conversation_history"] = [
        caller_message("Yes, that works."),
        agent_message("Great, see you then."),
    ]

    update = await closing_node(base_state)

    assert update["call_state"] == CallState.CLOSING
    assert update["call_summary"] == "Caller confirmed appointment for Oct 1 at 10 AM."
    assert any("Goodbye" in m["content"] for m in update["conversation_history"])
    assert fake_persister.completed, "call closure should be persisted"

    # Idempotent second run.
    base_state["metadata"].update(update["metadata"])
    assert await closing_node(base_state) == {}


async def test_escalation_marks_and_notifies(base_state, fake_persister):
    base_state["should_escalate"] = True
    update = await escalation_node(base_state)
    assert update["call_state"] == CallState.ESCALATION
    assert base_state["call_sid"] in fake_persister.escalated
    assert any("team" in m["content"] for m in update["conversation_history"])


# ── Conditional routing ──────────────────────────────────────────────────────


def test_route_prefers_escalation(base_state):
    base_state.update(
        should_escalate=True,
        caller_intent="cancellation",
        turn_count=1,
        conversation_history=[caller_message("manager please")],
    )
    assert route_after_response(base_state) == "escalate"


def test_route_closes_on_end_intent(base_state):
    base_state.update(
        should_escalate=False,
        caller_intent="end_call",
        turn_count=3,
        conversation_history=[caller_message("bye"), agent_message("bye!")],
    )
    assert route_after_response(base_state) == "close"


def test_route_continues_with_pending_caller(base_state):
    base_state.update(
        should_escalate=False,
        caller_intent="general_inquiry",
        turn_count=1,
        conversation_history=[caller_message("q1"), agent_message("a1"), caller_message("q2")],
    )
    assert route_after_response(base_state) == "continue"


def test_route_closes_when_turn_budget_exhausted(base_state):
    base_state.update(
        should_escalate=False,
        caller_intent="general_inquiry",
        turn_count=50,
        conversation_history=[caller_message("q"), agent_message("a")],
    )
    assert route_after_response(base_state) == "close"


def test_route_closes_after_consecutive_llm_failures(base_state):
    base_state.update(
        should_escalate=False,
        caller_intent="general_inquiry",
        turn_count=3,
        conversation_history=[caller_message("q"), agent_message("a")],
        metadata={"processed_upto": 1, "llm_failures": 3},
    )
    assert route_after_response(base_state) == "close"


# ── Compiled graph, end to end ───────────────────────────────────────────────


async def test_compiled_graph_processes_multi_turn_conversation(
    base_state, fake_responder, fake_memory_store, fake_persister
):
    fake_responder.replies = [
        "Your appointment is October 1st at 10 AM.",
        "Sure — October 3rd at 3 PM works.",
        "Caller rescheduled to Oct 3 at 3 PM.",
    ]
    base_state["conversation_history"] = [
        caller_message("Yes, speaking. What is this about?"),
        caller_message("Can I reschedule to the 3rd?"),
    ]

    agent = build_agent_graph()
    final = await agent.ainvoke(base_state)

    assert final["turn_count"] == 2
    assert final["caller_intent"] == "reschedule"
    assert final["call_summary"] == "Caller rescheduled to Oct 3 at 3 PM."
    assert final["call_state"] == CallState.CLOSING
    assert fake_persister.completed, "closing should persist the call"
    # Memory write-back of the resolved call.
    assert any(
        isinstance(r, MemoryRecordCreate) and "rescheduled" in r.content
        for r in fake_memory_store.upserted
    )


async def test_compiled_graph_escalates_then_closes(
    base_state, fake_responder, fake_memory_store, fake_persister
):
    fake_responder.replies = [
        "Let me look into that for you.",
        "Call escalated after complaint.",
    ]
    base_state["conversation_history"] = [
        caller_message("I have a complaint — get me a supervisor.")
    ]

    agent = build_agent_graph()
    final = await agent.ainvoke(base_state)

    assert final["should_escalate"] is True
    assert final["call_state"] == CallState.CLOSING
    assert base_state["call_sid"] in fake_persister.escalated
    contents = [m["content"] for m in final["conversation_history"]]
    assert any("hold" in c.lower() for c in contents)  # escalation hold message
