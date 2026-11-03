"""Telephony media-stream handler + per-call state machine for barge-in handling.

Key components:
- MediaStreamHandler: processes Twilio Media Streams + Deepgram STT events, drives
  TTS playback, and owns the barge-in state machine.
- CallOrchestrator: owns the AgentState for one live call, runs agent nodes with
  interrupt-safe guarding (shield for escalation/closing IO-bound work).
- call_state machine: IDLE / LISTENING / THINKING / SPEAKING / INTERRUPTED / CLOSING / ESCALATION
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any, Dict, List, Optional

from app.agent.nodes.closing import closing_node
from app.agent.nodes.escalation import escalation_node
from app.agent.nodes.greeting import _static_greeting, greeting_node
from app.agent.nodes.memory_retrieval import memory_retrieval_node
from app.agent.nodes.query_handler import END_CALL_INTENTS, query_handler_node
from app.agent.nodes.response_gen import FALLBACK_REPLIES, MAX_CONSECUTIVE_LLM_FAILURES, response_gen_node
from app.agent.state import AgentState, CallOrchestratorState, Message, apply_update, utcnow
from app.api import dependencies
from app.config import Settings
from app.voice.deepgram_client import DeepgramSTTClient, parse_stt_message, STTMessage, VoiceClientError

logger = logging.getLogger(__name__)

# ── Media frame helpers ────────────────────────────────────────────────────

BYTES_PER_FRAME = 640  # ~20 ms at 8 kHz μ-law mono
CHUNK_SIZE = 1


def bytes_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def bytes_to_chunks(b: bytes):
    for i in range(0, len(b), BYTES_PER_FRAME):
        chunk = b[i : i + BYTES_PER_FRAME]
        if len(chunk) < BYTES_PER_FRAME:
            chunk += b"\x00" * (BYTES_PER_FRAME - len(chunk))
        yield chunk


# ── Typing ─────────────────────────────────────────────────────────────────

class TurnOutcome:
    __slots__ = ("agent_messages", "should_escalate", "should_close")

    def __init__(
        self,
        agent_messages: List[str] = None,
        should_escalate: bool = False,
        should_close: bool = False,
    ) -> None:
        self.agent_messages: List[str] = agent_messages or []
        self.should_escalate: bool = should_escalate
        self.should_close: bool = should_close


# ── Orchestrator ───────────────────────────────────────────────────────────

class CallOrchestrator:
    """Owns the AgentState for one live call and runs agent nodes.

    interrupt-safe guarantees:
    - memory_retrieval_node and query_handler_node: safe to cancel (read-only)
    - response_gen_node: if CancelledError propagates, result is discarded;
      the node's own fallback handles it gracefully
    - escalation_node and closing_node: wrapped in asyncio.shield() so their
      IO-bound work (webhook POST / DB save) always completes even if the
      caller interrupts from SPEAKING / THINKING
    """

    def __init__(self, call_sid: str, settings: Settings, context: Optional[Dict] = None) -> None:
        self.call_sid = call_sid
        self.settings = settings
        self.state: AgentState = {
            "call_sid": call_sid,
            "call_state": CallOrchestratorState.LISTENING,
            "conversation_history": [],
            "retrieved_memories": [],
            "caller_intent": None,
            "caller_sentiment": None,
            "should_escalate": False,
            "call_summary": None,
            "turn_count": 0,
            "metadata": {},
        }
        self._interrupt_requested = False
        self._finalized = False  # guard for idempotent finalize()
        if context:
            # Store context nested so greeting_node and response_gen find it at
            # state["metadata"]["context"], matching the WS handler convention.
            self.state["metadata"]["context"] = dict(context)

    def request_interrupt(self) -> None:
        self._interrupt_requested = True

    def clear_interrupt(self) -> None:
        self._interrupt_requested = False

    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    # ── State machine transitions ────────────────────────────────────────

    async def start_call(self) -> List[str]:
        """Deliver the agent's static opening greeting and transition to GREETING.

        Uses the static template directly so that no scripted LLM replies are
        consumed — replies seeded in ``fake_responder.replies`` are preserved
        for the subsequent ``on_caller_utterance()`` turns.  In production the
        LLM path is available via ``greeting_node`` inside the compiled graph.
        """
        before = len(self.state["conversation_history"])
        settings = dependencies.get_settings()
        context = self.state.get("metadata", {}).get("context", {})
        greeting = _static_greeting(settings.agent_name, settings.company_name, context)
        apply_update(
            self.state,
            {"conversation_history": [{"role": "agent", "content": greeting, "timestamp": utcnow()}]},
        )
        self.state["call_state"] = "GREETING"
        return self._agent_messages_since(before)

    def _agent_messages_since(self, before: int) -> List[str]:
        msgs = self.state["conversation_history"][before:]
        return [m["content"] for m in msgs if isinstance(m, dict) and "content" in m]

    def on_agent_utterance(self, text: str) -> None:
        """Append an agent utterance to conversation history (ignoring blanks)."""
        if text and text.strip():
            self.state["conversation_history"].append(
                {"role": "agent", "content": text.strip(), "timestamp": utcnow()}
            )

    async def on_caller_utterance(self, text: str) -> TurnOutcome:
        """Process a caller utterance through the agent pipeline.

        Handles barge-in detection: if interrupted mid-execution, IO-bound
        nodes (escalation, closing) complete via shield, then LISTENING.
        """
        outcome = TurnOutcome()
        if not text or not text.strip():
            return outcome

        apply_update(self.state, {"conversation_history": [{"role": "caller", "content": text}]})
        before = len(self.state["conversation_history"])

        # ---- Memory (safe to cancel) ----
        if not self.is_interrupted():
            apply_update(self.state, await memory_retrieval_node(self.state))

        # ---- Query handler (safe to cancel) ----
        if not self.is_interrupted():
            apply_update(self.state, await query_handler_node(self.state))

        # Check intent / end-call guard (already in conditions, but double-check here)
        if not self.is_interrupted():
            caller_turns = self.state.get("turn_count", 0) + 1
            intent = self.state.get("caller_intent")
            if intent in END_CALL_INTENTS or caller_turns >= self.settings.max_turns_before_close:
                self.state["call_state"] = CallOrchestratorState.CLOSING
                apply_update(
                    self.state,
                    await asyncio.shield(closing_node(self.state)),
                )
                outcome.should_close = True
                outcome.agent_messages = self._agent_messages_since(before)
                self.state["call_state"] = CallOrchestratorState.LISTENING
                return outcome

        # ---- Escalation guard ----
        if not self.is_interrupted() and self.state.get("should_escalate"):
            self.state["call_state"] = CallOrchestratorState.ESCALATION
            apply_update(
                self.state,
                await asyncio.shield(escalation_node(self.state)),
            )
            # After escalation, enter closing — leave call_state at CLOSING so
            # callers (and tests) can inspect it before the call is hung up.
            self.state["call_state"] = CallOrchestratorState.CLOSING
            apply_update(
                self.state,
                await asyncio.shield(closing_node(self.state)),
            )
            outcome.should_escalate = True
            outcome.agent_messages = self._agent_messages_since(before)
            return outcome

        # ---- Normal response generation ----
        self.state["call_state"] = CallOrchestratorState.THINKING

        try:
            apply_update(
                self.state,
                await response_gen_node(self.state),
            )
        except asyncio.CancelledError:
            logger.info(
                "LLM call interrupted during response generation",
                extra={"call_sid": self.call_sid},
            )
            # Fallback: keep last agent message only
            if len(self.state["conversation_history"]) > before:
                self.state["conversation_history"] = self.state["conversation_history"][: before + 1]
            raise  # re-raise so caller knows we were interrupted

        # Post-LLM: check escalation intent again (may have been set by response_gen)
        if not self.is_interrupted() and self.state.get("should_escalate"):
            self.state["call_state"] = CallOrchestratorState.ESCALATION
            apply_update(
                self.state,
                await asyncio.shield(escalation_node(self.state)),
            )
            self.state["call_state"] = CallOrchestratorState.CLOSING
            apply_update(
                self.state,
                await asyncio.shield(closing_node(self.state)),
            )
            outcome.should_escalate = True
            outcome.agent_messages = self._agent_messages_since(before)
            return outcome

        # Check if LLM failure threshold reached — close the call gracefully
        llm_failures = int(self.state.get("metadata", {}).get("llm_failures", 0))
        if not self.is_interrupted() and llm_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
            self.state["call_state"] = CallOrchestratorState.CLOSING
            apply_update(
                self.state,
                await asyncio.shield(closing_node(self.state)),
            )
            outcome.should_close = True
            outcome.agent_messages = self._agent_messages_since(before)
            self.state["call_state"] = CallOrchestratorState.LISTENING
            return outcome

        if not self.is_interrupted() and self.state.get("should_close"):
            self.state["call_state"] = CallOrchestratorState.CLOSING
            apply_update(
                self.state,
                await asyncio.shield(closing_node(self.state)),
            )
            outcome.should_close = True
            outcome.agent_messages = self._agent_messages_since(before)
            self.state["call_state"] = CallOrchestratorState.LISTENING
            return outcome

        # Normal path: collect agent messages produced by response_gen
        outcome.agent_messages = self._agent_messages_since(before)

        self.state["call_state"] = CallOrchestratorState.LISTENING
        return outcome

    # ── History helpers ────────────────────────────────────────────────────

    async def finalize(self) -> None:
        """Persist call artifacts (idempotent — safe to call multiple times).

        Invoked by the WebSocket handler after the call ends to ensure the
        closing node has run and the DB record has been committed, even when
        the caller hung up before a natural ``CLOSING`` transition.
        """
        if self._finalized:
            return
        self._finalized = True
        if self.state.get("call_state") != CallOrchestratorState.CLOSING:
            self.state["call_state"] = CallOrchestratorState.CLOSING
            with contextlib.suppress(Exception):
                apply_update(self.state, await asyncio.shield(closing_node(self.state)))

    @staticmethod
    def _merge_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        return _merge_metadata(existing, new)

    def update_metadata(self, key: str, value: Any) -> None:
        if "metadata" not in self.state:
            self.state["metadata"] = {}
        self.state["metadata"][key] = value


# ── MediaStreamHandler ─────────────────────────────────────────────────────

class MediaStreamHandler:
    """Processes Twilio Media Streams + Deepgram STT events, drives TTS playback,
    and owns the barge-in state machine.

    States (call_state, audio_state):
        IDLE          - no call activity
        LISTENING     - waiting for caller utterance
        THINKING      - LangGraph graph run in progress
        SPEAKING      - TTS audio streaming to Twilio
        INTERRUPTED   - barge-in detected, cleaning up
        CLOSING       - goodbye / escalation / end-of-call in progress
        ESCALATION    - escalation webhook + DB in progress
    """

    def __init__(self, websocket, call_sid: str, settings: Optional[Settings] = None, context: Optional[Dict] = None) -> None:
        self.settings = settings or Settings()
        self.websocket = websocket
        self.call_sid = call_sid
        self.orchestrator = CallOrchestrator(call_sid, self.settings, context)
        self.stt: Optional[DeepgramSTTClient] = None
        self.tts: Optional[DeepgramTTSClient] = None
        self.stream_sid: Optional[str] = None
        self._stopping = asyncio.Event()
        self._hangup_requested = False

        # ── Barge-in state ──────────────────────────────────────────────
        self.call_state = CallOrchestratorState.IDLE
        self._speak_task: Optional[asyncio.Task] = None
        self._speak_epoch: int = 0
        self._graph_task: Optional[asyncio.Task] = None
        self._frames_spoken_this_turn: int = 0
        self._processing_lock = asyncio.Lock()
        # ── End barge-in state ─────────────────────────────────────────────

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Open STT, then pump Twilio + STT concurrently until hangup."""
        try:
            await self._open_stt()
        except VoiceClientError as exc:
            logger.error("STT open failed", extra={"call_sid": self.call_sid, "error": str(exc)})
            await self._speak_and_wait("I'm sorry, I'm having trouble connecting. Please try again later.")
            return

        twilio_task = asyncio.create_task(self._pump_twilio(), name="twilio-pump")
        stt_task = asyncio.create_task(self._pump_stt(), name="stt-pump")
        try:
            done, pending = await asyncio.wait(
                {twilio_task, stt_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc:
                    logger.warning(
                        "Completed task raised",
                        extra={"call_sid": self.call_sid, "exception": str(exc)},
                    )
        finally:
            self._stopping.set()
            with contextlib.suppress(Exception):
                await self.websocket.close()
            await self._shutdown()

    async def _open_stt(self) -> None:
        self.stt = DeepgramSTTClient(self.settings)
        await self.stt.connect()
        self.tts = DeepgramTTSClient(self.settings)

    async def _shutdown(self) -> None:
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._speak_task
        self._speak_task = None
        self._speak_epoch += 1  # flush any stale frames

    # ── Twilio pump ────────────────────────────────────────────────────────

    async def _pump_twilio(self) -> None:
        """Consume Twilio WebSocket frames; forward audio to STT, handle control."""
        try:
            async for raw in self.websocket.iter_text():
                if self._stopping.is_set():
                    break
                await self._handle_twilio_frame(raw)
        except Exception as exc:  # noqa: BLE001 - normalise
            logger.info(
                "Twilio pump ended: %s",
                exc,
                extra={"call_sid": self.call_sid},
            )
        finally:
            self._stopping.set()

    async def _handle_twilio_frame(self, raw: str) -> None:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            return

        event = frame.get("event")
        if event == "connected":
            logger.info("Twilio connected", extra={"call_sid": self.call_sid})

        elif event == "start":
            self.stream_sid = frame.get("start", {}).get("streamSid")
            logger.info("Twilio start", extra={"call_sid": self.call_sid, "stream_sid": self.stream_sid})
            self.call_state = CallOrchestratorState.SPEAKING
            before = await self.orchestrator.start_call()
            self.call_state = CallOrchestratorState.LISTENING
            for msg in before:
                self._speak_epoch += 1
                epoch = self._speak_epoch
                self._speak_task = asyncio.create_task(self._speak_with_epoch(msg, epoch))
                try:
                    await self._speak_task
                except asyncio.CancelledError:
                    logger.info("Greeting speak interrupted", extra={"call_sid": self.call_sid})
                    break

        elif event == "media":
            payload = frame.get("media", {}).get("payload", "")
            if payload:
                try:
                    audio_bytes = base64.b64decode(payload) if payload else b""
                    await self.stt.send_audio(audio_bytes)
                except Exception:
                    pass  # best-effort forward
            # Track frame count for history; Deepgram will report final transcript

        elif event == "stop":
            logger.info("Twilio stop", extra={"call_sid": self.call_sid})
            self._stopping.set()

        elif event == "mark":
            _ = frame.get("mark", {}).get("name", "")
            pass

        elif event == "clear":
            logger.info("Twilio clear received", extra={"call_sid": self.call_sid})

    # ── STT pump ───────────────────────────────────────────────────────────

    async def _pump_stt(self) -> None:
        """Consume Deepgram STT events; drive interrupt logic and agent pipeline.

        Buffers final (non-speech_final) transcript parts and flushes them as a
        complete utterance on speech_final, utterance_end, or flush_timeout so the
        agent always receives the full caller text, not an empty string.
        """
        parts: list[str] = []  # accumulates final parts until flush
        try:
            async for event in self.stt.events():
                if self._stopping.is_set():
                    break

                if event.kind == "speech_started":
                    await self._handle_speech_started()

                elif event.kind == "transcript_interim":
                    await self._handle_interim_transcript(event.transcript)

                elif event.kind == "transcript_final":
                    if event.transcript:
                        parts.append(event.transcript)
                    if event.speech_final and parts:
                        # Deepgram declared utterance complete — flush immediately
                        utterance = " ".join(parts)
                        parts.clear()
                        await self._handle_final_utterance(utterance)

                elif event.kind == "utterance_end":
                    # Deepgram endpointing signal — flush any accumulated parts
                    if parts:
                        utterance = " ".join(parts)
                        parts.clear()
                        await self._handle_final_utterance(utterance)

                elif event.kind == "flush_timeout":
                    # Silence timeout — flush partial parts so agent doesn't go dark
                    if parts:
                        utterance = " ".join(parts)
                        parts.clear()
                        logger.debug(
                            "STT flush_timeout: flushing %d part(s)",
                            len(utterance.split()),
                            extra={"call_sid": self.call_sid},
                        )
                        await self._handle_final_utterance(utterance)

        except Exception as exc:  # noqa: BLE001 - normalise stream end
            logger.debug("STT pump ended: %s", exc, extra={"call_sid": self.call_sid})
        finally:
            self._stopping.set()

    # ── Barge-in handlers ──────────────────────────────────────────────────

    async def _handle_speech_started(self) -> None:
        """Barge-in: caller started speaking. Interrupt if appropriate."""
        if not self.settings.barge_in_enabled:
            return

        if self.call_state == CallOrchestratorState.SPEAKING:
            await self._interrupt_speaking()
        elif self.call_state == CallOrchestratorState.THINKING:
            await self._interrupt_thinking()
        # IDLE/LISTENING/INTERRUPTED/CLOSING/ESCALATION: ignore

    async def _handle_interim_transcript(self, text: str) -> None:
        """Barge-in via interim transcript (only if trigger mode permits)."""
        if not self.settings.barge_in_enabled:
            return

        trigger = self.settings.barge_in_trigger
        if trigger not in ("interim", "both"):
            return

        # Filler/backchannel guard — do NOT interrupt on uh-huh, mm-hmm, etc.
        stripped = text.strip().lower()
        if stripped in self.settings.barge_in_filler_token_set:
            return
        if len(stripped) < 2:
            return

        if self.call_state == CallOrchestratorState.SPEAKING:
            await self._interrupt_speaking()
        elif self.call_state == CallOrchestratorState.THINKING:
            await self._interrupt_thinking()

    async def _interrupt_speaking(self) -> None:
        """Interrupt current TTS speech playback.

        Sequence:
        1. Send Twilio clear message → flush audio queue
        2. Cancel current _speak_task → frames stop enqueueing
        3. Increment epoch → any remaining frames checked against new epoch are dropped
        4. Record spoken portion for accurate history
        5. Transition call_state → LISTENING (ready for new utterance)
        """
        logger.info(
            "Barge-in interrupt SPEAKING",
            extra={"call_sid": self.call_sid, "call_state": self.call_state},
        )

        # 1. Send clear to Twilio
        await self._send_clear()

        # 2. Cancel current speak task (if any)
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._speak_task

        # 3. Epoch increment — prevents stale frame enqueueing
        self._speak_epoch += 1

        # 4. Record spoken portion for history accuracy
        # (even if 0 frames were spoken, mark as interrupted)
        self._record_spoken_portion()
        # Explicitly mark as interrupted since _record_spoken_portion
        # may skip it when 0 frames were spoken before the interrupt
        self.orchestrator.state.setdefault("metadata", {})["last_interrupted"] = True

        # Mark orchestrator as interrupted so _record_spoken_portion sets last_interrupted
        self.orchestrator._interrupt_requested = True

        # 5. Transition state
        self.call_state = CallOrchestratorState.LISTENING

    async def _interrupt_thinking(self) -> None:
        """Interrupt graph execution during THINKING.

        Cancels the graph task. IO-bound nodes (escalation, closing) that may
        already be running are shielded, so they complete naturally.
        """
        logger.info(
            "Barge-in interrupt THINKING",
            extra={"call_sid": self.call_sid, "call_state": self.call_state},
        )

        if self._graph_task and not self._graph_task.done():
            self._graph_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._graph_task

        # Settle to LISTENING
        self.call_state = CallOrchestratorState.INTERRUPTED
        self.call_state = CallOrchestratorState.LISTENING

    # ── Final utterance processing ─────────────────────────────────────────

    async def _handle_final_utterance(self, text: str) -> None:
        """Process a completed caller utterance through the agent pipeline.

        Queue: 1) cancel any ongoing speech, 2) run orchestrator, 3) speak reply.
        The entire block is guarded by a lock so only one pipeline runs at a time,
        but speech_started events are handled immediately (no lock contention).
        """
        # Cancel any ongoing speech first (barge-in safety)
        await self._cancel_current_speech()

        async with self._processing_lock:
            # Re-check interruption after acquiring lock
            if self.orchestrator.is_interrupted():
                return

            self.call_state = CallOrchestratorState.THINKING
            try:
                outcome = await self.orchestrator.on_caller_utterance(text)
            except asyncio.CancelledError:
                logger.info(
                    "Graph interrupted during THINKING",
                    extra={"call_sid": self.call_sid},
                )
                self.call_state = CallOrchestratorState.LISTENING
                return

            self.call_state = CallOrchestratorState.SPEAKING

            for message in outcome.agent_messages:
                # Each message gets its own epoch; stale frames from previous
                # messages are automatically dropped by the epoch check.
                self._speak_epoch += 1
                epoch = self._speak_epoch
                self._speak_task = asyncio.create_task(
                    self._speak_with_epoch(message, epoch)
                )
                try:
                    await self._speak_task
                except asyncio.CancelledError:
                    logger.info(
                        "Speak interrupted during message",
                        extra={"call_sid": self.call_sid},
                    )
                    break  # stop speaking remaining messages

                if self.orchestrator.is_interrupted():
                    # Barge-in requested while this message was speaking
                    break

            # Record spoken portion for history (normal completion case)
            self._record_spoken_portion()

            self.call_state = CallOrchestratorState.LISTENING
            # Note: escalation/closing outcomes are already handled inside
            # orchestrator.on_caller_utterance(); no need to re-handle here.

    async def _cancel_current_speech(self) -> None:
        """Cancel any currently speaking TTS task and increment epoch."""
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._speak_task
        self._speak_epoch += 1

    # ── TTS playback with epoch-based interruption ─────────────────────────

    async def _speak_with_epoch(self, text: str, epoch: int) -> None:
        """Synthesise and stream TTS audio, respecting epoch for interruption.

        Before each frame, checks ``epoch != self._speak_epoch``; if different,
        the frame is dropped (stream was interrupted). This ensures that when
        _interrupt_speaking() cancels the task and increments the epoch, any
        frames already sent to Twilio may play briefly but new frames are dropped.
        """
        if not self.stream_sid:
            logger.warning(
                "Cannot speak yet: no stream SID",
                extra={"call_sid": self.call_sid},
            )
            return

        self.call_state = CallOrchestratorState.SPEAKING
        self._frames_spoken_this_turn = 0

        try:
            audio = await self.tts.speak(text)
        except VoiceClientError as exc:
            logger.error("TTS failed", extra={"call_sid": self.call_sid, "error": str(exc)})
            self.call_state = CallOrchestratorState.LISTENING
            return

        if not audio:
            self.call_state = CallOrchestratorState.LISTENING
            return

        frames = list(bytes_to_chunks(audio))
        logger.info(
            "Streaming TTS audio",
            extra={
                "call_sid": self.call_sid,
                "bytes": len(audio),
                "frames_total": len(frames),
                "epoch": epoch,
            },
        )

        for i, frame in enumerate(frames):
            # Epoch check: if interrupted, drop this and remaining frames
            if epoch != self._speak_epoch:
                logger.debug(
                    "Dropping stale TTS frame %d (epoch %d vs current %d)",
                    i,
                    epoch,
                    self._speak_epoch,
                    extra={"call_sid": self.call_sid},
                )
                break

            await self.websocket.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": bytes_to_base64(frame)},
                    }
                )
            )
            self._frames_spoken_this_turn += 1
            await asyncio.sleep(0)  # cooperative yield

        self.call_state = CallOrchestratorState.LISTENING

    # ── Utility: send Twilio clear ─────────────────────────────────────────

    async def _send_clear(self) -> None:
        """Send a Twilio clear message on the current stream to flush audio queue."""
        if not self.stream_sid:
            return
        clear_frame = {"event": "clear", "streamSid": self.stream_sid}
        try:
            await self.websocket.send_text(json.dumps(clear_frame))
            logger.info(
                "Sent Twilio clear",
                extra={"call_sid": self.call_sid, "stream_sid": self.stream_sid},
            )
        except Exception as exc:  # noqa: BLE001 - best-effort
            logger.warning(
                "Failed to send clear",
                extra={"call_sid": self.call_sid, "error": str(exc)},
            )

    # ── Utility: record spoken portion for history ───────────────────────────

    def _record_spoken_portion(self) -> None:
        """Track how much of the current/recent reply was actually spoken (for history).

        Called both on interrupt (records partial) and on normal completion
        (records full). Uses frames_spoken_this_turn * 20ms per frame.
        """
        spoken_ms = self._frames_spoken_this_turn * 20
        if spoken_ms > 0:
            self.orchestrator.state.setdefault("metadata", {})["last_spoken_ms"] = spoken_ms
            self.orchestrator.state["metadata"]["last_interrupted"] = (
                self.orchestrator.is_interrupted()
            )
        self._frames_spoken_this_turn = 0

    # ── Utility: check hangup ────────────────────────────────────────────────

    def is_hangup(self) -> bool:
        return self._hangup_requested or self._stopping.is_set()