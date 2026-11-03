"""Tests for the live-call orchestration layer (no network, no WebSockets)."""

from __future__ import annotations

from app.telephony.media_stream import CallOrchestrator

from .conftest import caller_message


async def test_start_call_returns_greeting(base_state, fake_responder):
    from app.api import dependencies

    orchestrator = CallOrchestrator(
        "CA_TEST_123",
        dependencies.get_settings(),
        context={"customer_name": "Priya Sharma", "purpose": "appointment_reminder"},
    )
    greetings = await orchestrator.start_call()

    assert len(greetings) == 1
    assert "Priya Sharma" in greetings[0]
    assert orchestrator.state["call_state"] == "GREETING"


async def test_caller_utterance_runs_full_turn_pipeline(
    base_state, fake_memory_store, fake_responder
):
    """Memory retrieval → intent → Groq reply, all in one live turn."""
    from types import SimpleNamespace

    fake_memory_store.results = [
        SimpleNamespace(content="Free rescheduling within 24 hours.", distance=0.1)
    ]
    fake_responder.replies = ["Yes — rescheduling is free up to 24 hours ahead."]
    from app.api import dependencies

    orchestrator = CallOrchestrator("CA_TEST_123", dependencies.get_settings())
    await orchestrator.start_call()

    outcome = await orchestrator.on_caller_utterance("Can I reschedule my appointment?")

    assert outcome.should_escalate is False
    assert outcome.should_close is False
    assert orchestrator.state["caller_intent"] == "reschedule"
    # The LLM reply was composed and queued to be spoken via TTS.
    assert outcome.agent_messages == ["Yes — rescheduling is free up to 24 hours ahead."]
    assert orchestrator.state["turn_count"] == 1
    # Retrieved memory was injected into the responder's system prompt.
    assert "Free rescheduling" in fake_responder.calls[0]["system_prompt"]


async def test_escalation_utterance_triggers_escalation_and_close(
    base_state, fake_persister, fake_responder
):
    from app.api import dependencies

    fake_responder.replies = ["Call escalated per caller request."]
    orchestrator = CallOrchestrator("CA_TEST_123", dependencies.get_settings())

    outcome = await orchestrator.on_caller_utterance(
        "This is unacceptable, put a manager on right now."
    )

    assert outcome.should_escalate is True
    assert orchestrator.state["should_escalate"] is True
    assert orchestrator.state["call_state"] == "CLOSING"
    assert "CA_TEST_123" in fake_persister.escalated
    # Escalation hold message + closing farewell were queued to speak.
    assert any("hold" in m.lower() for m in outcome.agent_messages)


async def test_end_call_intent_closes_and_persists_once(base_state, fake_persister, fake_responder):
    from app.api import dependencies

    fake_responder.replies = ["Caller completed rescheduling to Oct 3."]
    orchestrator = CallOrchestrator("CA_TEST_123", dependencies.get_settings())

    outcome = await orchestrator.on_caller_utterance("No thanks, that's all.")
    assert outcome.should_close is True
    assert orchestrator.state["call_summary"] == "Caller completed rescheduling to Oct 3."

    await orchestrator.finalize()
    await orchestrator.finalize()  # idempotent

    assert len(fake_persister.completed) == 1
    completed = fake_persister.completed[0]
    assert completed["summary"] == "Caller completed rescheduling to Oct 3."
    roles = [m["role"] for m in completed["history"]]
    assert "caller" in roles and "agent" in roles


async def test_orchestrator_closes_after_repeated_llm_failures(base_state, fake_persister, fake_responder):
    """A dead LLM must end the call politely instead of parroting forever."""
    from app.agent.nodes.response_gen import FALLBACK_REPLIES
    from app.api import dependencies

    class FailingResponder:
        async def __call__(self, *, system_prompt, history):
            raise RuntimeError("groq down")

    # The failing responder is also used by closing_node's summary, which
    # itself falls back to a deterministic summary — safe.
    dependencies.set_override("response_generator", FailingResponder())

    orchestrator = CallOrchestrator("CA_TEST_123", dependencies.get_settings())

    outcomes = [
        await orchestrator.on_caller_utterance("hello?")
        for _ in range(3)
    ]

    assert outcomes[0].should_close is False
    assert outcomes[1].should_close is False
    assert outcomes[2].should_close is True, "3rd consecutive LLM failure must close the call"
    # Varied fallbacks spoken, then the closing farewell.
    spoken = [m for o in outcomes for m in o.agent_messages]
    assert spoken[0] == FALLBACK_REPLIES[0]
    assert spoken[1] == FALLBACK_REPLIES[1]
    assert spoken[2] == FALLBACK_REPLIES[2]
    assert "Goodbye" in spoken[-1]
    assert fake_persister.completed, "closing artefacts must persist"


def test_agent_utterance_recorded_in_history(base_state):
    from app.api import dependencies

    orchestrator = CallOrchestrator("CA_TEST_123", dependencies.get_settings())
    orchestrator.on_agent_utterance("Of course, let me check that for you.")
    orchestrator.on_agent_utterance("")  # blank transcripts are ignored

    history = orchestrator.state["conversation_history"]
    assert len(history) == 1
    assert history[0]["role"] == "agent"
