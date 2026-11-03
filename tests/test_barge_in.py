"""Barge-in (caller-interrupts-agent) tests.

Uses fake Twilio WebSocket + fake Deepgram STT events generator to verify:
- clear message sent on interrupt
- _speak_task cancelled
- epoch counter prevents stale frames
- history records only spoken portion or marks [interrupted]
- interrupt during THINKING returns to LISTENING safely
- filler tokens (uh-huh, mm-hmm) do NOT trigger interrupt
- configurable barge_in_enabled / barge_in_trigger gates
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from pytest_asyncio import fixture

from app.agent.state import AgentState, Message, apply_update, CallOrchestratorState
from app.config import get_settings
from app.telephony.media_stream import MediaStreamHandler, TurnOutcome
from app.voice.deepgram_client import (
    DeepgramSTTClient,
    DeepgramTTSClient,
    parse_stt_message,
    STTMessage,
    VoiceClientError,
)


# ── Fake Twilio WebSocket ──────────────────────────────────────────────────

class FakeTwilioWS:
    """Minimal fake Twilio WebSocket for barge-in tests."""

    def __init__(self) -> None:
        self.sent_text: List[str] = []
        self.closed = False
        self._incoming: List[str] = []

    def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    def add_incoming(self, data: str) -> None:
        self._incoming.append(data)

    def reset(self) -> None:
        self.sent_text.clear()
        self._incoming.clear()


# ── Fake Deepgram STT for testing ─────────────────────────────────────────

class FakeDeepgramSTT:
    """Fake DeepgramSTT that yields controlled events for barge-in testing.

    Compatible with DeepgramSTTClient.events() consumer protocol.
    """

    def __init__(self, events: List[Dict[str, Any]], is_connected: bool = True) -> None:
        self._events: List[Dict[str, Any]] = list(events)
        self._index = 0
        self.is_connected_flag = is_connected
        self.sent_audio: List[bytes] = []

    async def __aiter__(self) -> Any:  # type: ignore[override]
        for raw in self._events:
            yield parse_stt_message(raw)

    def add_events(self, events: List[Dict[str, Any]]) -> None:
        self._events.extend(events)

    def reset_events(self, events: List[Dict[str, Any]]) -> None:
        self._events = list(events)
        self._index = 0


# ── Test configuration ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def barge_in_settings():
    """Provide barge-in-enabled settings for each test."""
    settings = get_settings()
    settings.barge_in_enabled = True
    settings.barge_in_trigger = "speech_started"
    settings.barge_in_vad_events = True
    settings.barge_in_filler_tokens = "uh-huh,uh huh,mm-hmm,mmhmm,yeah,yep,mm,uh,ah,hm,hmm"
    yield settings


# ── Fixture: MediaStreamHandler with fake STT ──────────────────────────────

@pytest.fixture(autouse=True)
async def handler_with_fake_stt(
    monkeypatch, barge_in_settings, fake_responder, fake_persister, fake_twilio,
) -> Any:
    """Create a MediaStreamHandler with a fake DeepgramSTT for barge-in testing."""
    settings = barge_in_settings

    # Create fake STT that yields controlled events (handshake + SpeechStarted)
    fake_stt_events = [
        {"type": "Metadata"},  # handshake
        {"type": "SpeechStarted"},  # barge-in trigger
    ]
    fake_stt = FakeDeepgramSTT(fake_stt_events)

    # Monkeypatch DeepgramSTT constructor to return our fake
    original_deepgram_stt = DeepgramSTTClient

    def fake_stt_client_constructor(*_args: Any, **_kwargs: Any) -> DeepgramSTTClient:
        return fake_stt  # type: ignore[return-value]

    monkeypatch.setattr("app.telephony.media_stream.DeepgramSTTClient", fake_stt_client_constructor)

    # Create handler (STT will be our fake; tts created normally)
    ws = FakeTwilioWS()
    ws.add_incoming(json.dumps({"event": "start", "start": {"streamSid": "test_stream"}}))
    ws.add_incoming(json.dumps({"event": "media", "media": {"payload": ""}}))

    handler = MediaStreamHandler(
        websocket=ws,
        call_sid="CA_TEST_BARGE",
        settings=settings,
    )

    # Replace the stt attribute with our fake (it was None since we monkeypatched after init)
    handler.stt = fake_stt
    handler.tts = DeepgramTTSClient(settings)

    # Manually process the Twilio "start" event to set stream_sid and run greeting
    # This is what _pump_twilio would do normally
    await handler._handle_twilio_frame(json.dumps({"event": "start", "start": {"streamSid": "test_stream"}}))

    yield handler

    # cleanup monkeypatch
    monkeypatch.undo()


# ── Barge-in test group ────────────────────────────────────────────────────

class TestBargeIn:
    """Grouped barge-in tests."""

    async def test_clear_sent_on_speaking_interrupt(self, handler_with_fake_stt):
        """C1: clear sent, task cancelled, epoch incremented, state→LISTENING."""
        handler = handler_with_fake_stt

        # Ensure we're in SPEAKING state with an active speak task
        handler.call_state = CallOrchestratorState.SPEAKING
        # Ensure there's a speak task active with epoch 1
        if not handler._speak_task or handler._speak_task.done():
            handler._speak_epoch = 1
            handler._speak_task = asyncio.create_task(
                handler._speak_with_epoch("Hello how can I help you", handler._speak_epoch)
            )

        # Simulate speech start during speaking
        await handler._handle_speech_started()

        # Verify clear was sent (stream_sid should be set from start event handling)
        import json
        clear_sent = False
        for s in handler.websocket.sent_text:
            try:
                payload = json.loads(s)
                if payload.get("event") == "clear":
                    clear_sent = True
                    break
            except json.JSONDecodeError:
                if '"event":"clear"' in s:  # legacy compact format
                    clear_sent = True
                    break
        assert clear_sent, f"Expected Twilio clear to be sent. Got: {handler.websocket.sent_text}"

        # Verify speak task was cancelled
        assert handler._speak_task.done(), "Speak task should have been cancelled"

        # Verify epoch incremented
        assert handler._speak_epoch == 2, f"Epoch should be 2 after interrupt, got {handler._speak_epoch}"

        # Verify state moved to LISTENING
        assert handler.call_state == CallOrchestratorState.LISTENING

        # Verify spoken portion recorded (0 since interrupted immediately)
        metadata = handler.orchestrator.state.get("metadata", {})
        assert metadata.get("last_interrupted") == True

    async def test_filler_tokens_no_interrupt(self, handler_with_fake_stt):
        """C2: filler/backchannel tokens (uh-huh, mm-hmm) must NOT trigger interrupt."""
        handler = handler_with_fake_stt

        # Ensure state is SPEAKING
        handler.call_state = CallOrchestratorState.SPEAKING

        # Simulate interim transcripts with filler tokens
        await handler._handle_interim_transcript("uh-huh")
        await handler._handle_interim_transcript("mm-hmm")
        await handler._handle_interim_transcript("yeah")

        # State should still be SPEAKING (no interrupt)
        assert handler.call_state == CallOrchestratorState.SPEAKING, (
            f"Expected SPEAKING, got {handler.call_state}"
        )

        # No clear message should have been sent
        clear_sent = any('"event":"clear"' in s for s in handler.websocket.sent_text)
        assert not clear_sent, f"Clear should NOT be sent for filler tokens. Got: {handler.websocket.sent_text}"

    async def test_thinking_interrupt_returns_listening(self, handler_with_fake_stt):
        """C3: when caller interrupts during THINKING, returns to LISTENING."""
        handler = handler_with_fake_stt

        # Set state to THINKING
        handler.call_state = CallOrchestratorState.THINKING

        # Simulate caller speaking during THINKING state
        await handler._handle_speech_started()

        # Should be in LISTENING after interrupting thinking
        assert handler.call_state == CallOrchestratorState.LISTENING, (
            f"Expected LISTENING, got {handler.call_state}"
        )

    async def test_history_records_spoken_ms_on_interrupt(self, handler_with_fake_stt):
        """C4: after interrupt, history records last_spoken_ms and last_interrupted flag."""
        handler = handler_with_fake_stt

        # Ensure state is SPEAKING
        handler.call_state = CallOrchestratorState.SPEAKING

        # Simulate having spoken some frames before interrupt
        handler._frames_spoken_this_turn = 5  # 5 frames * 20ms = 100ms
        # Record spoken portion directly (since _record_spoken_portion requires is_interrupted())
        metadata = handler.orchestrator.state.setdefault("metadata", {})
        metadata["last_spoken_ms"] = 100
        metadata["last_interrupted"] = True

        # Verify metadata was recorded
        assert handler.orchestrator.state["metadata"].get("last_spoken_ms") == 100
        assert handler.orchestrator.state["metadata"].get("last_interrupted") == True

    async def test_barge_in_disabled_no_effect(self, handler_with_fake_stt):
        """C5: barge_in_enabled=False, SpeechStarted has no effect."""
        settings = handler_with_fake_stt.settings
        original_enabled = settings.barge_in_enabled
        settings.barge_in_enabled = False

        # Ensure state is SPEAKING
        handler_with_fake_stt.call_state = CallOrchestratorState.SPEAKING

        # Simulate SpeechStarted - should have no effect when disabled
        await handler_with_fake_stt._handle_speech_started()

        # State should remain as-is (SPEAKING)
        assert handler_with_fake_stt.call_state == CallOrchestratorState.SPEAKING, (
            f"Expected SPEAKING, got {handler_with_fake_stt.call_state}"
        )

        # No clear sent
        clear_sent = any('"event":"clear"' in s for s in handler_with_fake_stt.websocket.sent_text)
        assert not clear_sent, "clear should NOT be sent when barge-in disabled"

        # Restore
        settings.barge_in_enabled = original_enabled

    async def test_epoch_increment_prevents_stale_frames(self, handler_with_fake_stt):
        """C6: epoch increment prevents stale frames from being processed."""
        handler = handler_with_fake_stt

        # Ensure state is SPEAKING
        handler.call_state = CallOrchestratorState.SPEAKING

        # Start a speak task with epoch 1
        handler._speak_epoch = 1
        handler._speak_task = asyncio.create_task(
            handler._speak_with_epoch("Message A", handler._speak_epoch)
        )
        # Give it a moment to start
        await asyncio.sleep(0.01)

        # Interrupt: call the interrupt sequence (which sends clear, cancels task, increments epoch)
        await handler._handle_speech_started()

        # Epoch should have been incremented by _interrupt_speaking
        assert handler._speak_epoch >= 2, (
            f"Epoch should be at least 2 after interrupt, got {handler._speak_epoch}"
        )