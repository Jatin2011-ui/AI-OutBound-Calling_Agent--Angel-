"""Deepgram clients: streaming speech-to-text (WebSocket) and text-to-speech (REST).

Both are tuned for Twilio Media Streams — 8 kHz, mono, G.711 μ-law audio —
so raw Twilio frames can be forwarded to STT untouched and TTS output can be
streamed straight back without transcoding.

STT robustness guarantees:

- ``connect()`` validates the handshake: Deepgram rejects bad keys/params
  either during the HTTP upgrade *or* as an immediate ``{"type": "Error"}``
  frame. Both are surfaced as :class:`VoiceClientError` instead of hanging.
- ``events()`` yields every Deepgram frame as a typed :class:`STTMessage`,
  including ``SpeechStarted`` (requires ``vad_events=true``), interim
  transcripts (requires ``interim_results=true``), final transcripts, and
  ``UtteranceEnd``.
- ``utterances()`` remains available for batch-style processing that only
  needs complete, endpointed utterances.
- Verbose debug logging at every hop so a silent pipeline is impossible to
  miss in the logs.

TTS: ``POST https://api.deepgram.com/v1/speak?...&encoding=mulaw`` returns
raw μ-law bytes, ready to chunk into 20 ms Twilio media frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.config import Settings

try:  # websockets >= 12 (new asyncio client)
    from websockets.asyncio.client import connect as _ws_connect

    _HEADERS_KWARG = "additional_headers"
except ImportError:  # pragma: no cover - legacy websockets fallback
    from websockets.legacy.client import connect as _ws_connect  # type: ignore[no-redef]

    _HEADERS_KWARG = "extra_headers"

logger = logging.getLogger(__name__)

_KEEPALIVE_INTERVAL_S = 5.0


class VoiceClientError(Exception):
    """Raised when a Deepgram connection or request fails."""


@dataclass
class STTMessage:
    """A parsed Deepgram STT event.

    ``kind`` is one of:

    - ``"speech_started"`` — Deepgram detected voice activity (VAD).
    - ``"transcript_interim"`` — non-final partial transcript.
    - ``"transcript_final"`` — finalised transcript segment.
    - ``"utterance_end"`` — Deepgram declared the utterance complete.
    - ``"flush_timeout"`` — ``flush_timeout`` elapsed with no new frame; buffered parts should be flushed.
    - ``"other"`` — metadata, keep-alive, or unrecognised frame.
    """

    kind: str
    transcript: str = ""
    is_final: bool = False
    speech_final: bool = False


def parse_stt_message(raw: Dict[str, Any]) -> STTMessage:
    """Parse one Deepgram ``listen`` WebSocket JSON frame into an STTMessage."""
    msg_type = raw.get("type", "")

    if msg_type == "SpeechStarted":
        return STTMessage(kind="speech_started")

    if msg_type == "Results":
        alternatives = (raw.get("channel") or {}).get("alternatives") or []
        transcript = ""
        if alternatives and isinstance(alternatives[0], dict):
            transcript = str(alternatives[0].get("transcript") or "").strip()
        is_final = bool(raw.get("is_final"))
        speech_final = bool(raw.get("speech_final"))
        kind = "transcript_final" if is_final else "transcript_interim"
        return STTMessage(
            kind=kind,
            transcript=transcript,
            is_final=is_final,
            speech_final=speech_final,
        )

    if msg_type == "UtteranceEnd":
        return STTMessage(kind="utterance_end")

    return STTMessage(kind="other")


class DeepgramSTTClient:
    """Streaming speech-to-text over a Deepgram ``listen`` WebSocket.

    ``handshake_timeout`` bounds the post-connect peek that detects an
    immediate error frame. ``flush_timeout`` bounds how long a partial
    utterance may sit un-flushed when ``speech_final`` never arrives.
    """

    def __init__(
        self,
        settings: Settings,
        handshake_timeout: float = 3.0,
        flush_timeout: float = 2.0,
    ) -> None:
        self._settings = settings
        self._handshake_timeout = handshake_timeout
        self._flush_timeout = flush_timeout
        self._ws: Any = None
        self._keepalive_task: Optional[asyncio.Task] = None
        # Frames consumed during handshake validation are replayed to the
        # event consumer, in arrival order.
        self._pending: asyncio.Queue[Any] = asyncio.Queue()

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    async def connect(self) -> None:
        """Open the STT WebSocket and validate the handshake."""
        url = self._settings.deepgram_stt_listen_url
        headers = {"Authorization": f"Token {self._settings.deepgram_api_key}"}
        logger.info("Connecting Deepgram STT", extra={"stt_url": url})
        try:
            self._ws = await _ws_connect(
                url,
                **{_HEADERS_KWARG: headers},
                ping_interval=20,
                ping_timeout=20,
                max_size=4 * 1024 * 1024,
            )
        except Exception as exc:  # noqa: BLE001 - normalise all ws errors
            raise VoiceClientError(f"Deepgram STT connection failed: {exc}") from exc

        # Handshake validation: Deepgram either rejects the HTTP upgrade
        # (caught above) or sends an immediate Error frame. Peek briefly.
        try:
            first = await asyncio.wait_for(self._ws.recv(), timeout=self._handshake_timeout)
        except asyncio.TimeoutError:
            first = None  # No immediate error frame — connection looks good.
        except Exception as exc:  # noqa: BLE001
            await self.close()
            raise VoiceClientError(
                f"Deepgram STT socket closed during handshake: {exc}"
            ) from exc

        if first is not None:
            logger.debug("Deepgram STT first frame: %s", first)
            payload: Dict[str, Any] = {}
            if isinstance(first, str):
                with contextlib.suppress(json.JSONDecodeError):
                    payload = json.loads(first)
            if payload.get("type") == "Error":
                await self.close()
                raise VoiceClientError(f"Deepgram rejected STT connection: {first}")
            # Legitimately received event (e.g. Metadata) — replay to consumer.
            self._pending.put_nowait(first)

        self._keepalive_task = asyncio.create_task(self._keepalive())
        logger.info(
            "Deepgram STT connected",
            extra={"model": self._settings.deepgram_stt_model},
        )

    async def send_audio(self, mulaw_chunk: bytes) -> None:
        """Send a raw μ-law audio frame to the STT stream."""
        if self._ws is None:
            logger.debug("Dropping audio chunk: STT not connected")
            return
        try:
            await self._ws.send(mulaw_chunk)
        except Exception as exc:  # noqa: BLE001 - log; next chunk may recover
            logger.warning("Failed to send audio to Deepgram", extra={"error": str(exc)})

    async def _recv_message(self) -> Any:
        """Next raw frame: replayed handshake frames first, then live socket."""
        if not self._pending.empty():
            return self._pending.get_nowait()
        return await asyncio.wait_for(self._ws.recv(), timeout=self._flush_timeout)

    async def events(self) -> AsyncGenerator[STTMessage, None]:
        """Yield every Deepgram frame as a typed :class:`STTMessage`.

        Includes ``speech_started`` (VAD), ``transcript_interim``,
        ``transcript_final``, and ``utterance_end``. Use this for barge-in
        detection where you need the earliest possible signal.
        """
        while self._ws is not None:
            try:
                raw = await self._recv_message()
            except asyncio.TimeoutError:
                # flush_timeout elapsed with no new frame — signal consumers to
                # flush any buffered partial utterance before waiting again.
                logger.debug("Deepgram STT flush_timeout: no frame received")
                yield STTMessage(kind="flush_timeout")
                continue
            except Exception as exc:  # noqa: BLE001 - any close ends the stream
                logger.debug("Deepgram STT stream ended: %s", exc)
                break

            if not isinstance(raw, str):
                logger.debug("Binary frame from Deepgram ignored (%d bytes)", len(raw))
                continue
            logger.debug("Deepgram STT message: %s", raw)
            try:
                message = parse_stt_message(json.loads(raw))
            except json.JSONDecodeError:
                continue

            if message.kind in ("transcript_final",) and message.transcript:
                logger.info(
                    "STT final transcript",
                    extra={"transcript": message.transcript},
                )
            yield message

    async def utterances(self) -> AsyncGenerator[str, None]:
        """Yield complete caller utterances (batch-style convenience).

        Segments on Deepgram endpointing: buffered final transcript parts
        flush on ``speech_final`` or ``UtteranceEnd``. Falls back to a
        timeout flush when ``speech_final`` never arrives.
        """
        parts: List[str] = []
        async for message in self.events():
            if message.kind == "transcript_final":
                if message.transcript:
                    parts.append(message.transcript)
                if message.speech_final and parts:
                    utterance = " ".join(parts)
                    parts.clear()
                    logger.info("STT utterance (speech_final)", extra={"utterance": utterance})
                    yield utterance
            elif message.kind == "utterance_end":
                if parts:
                    utterance = " ".join(parts)
                    parts.clear()
                    logger.info("STT utterance (utterance_end)", extra={"utterance": utterance})
                    yield utterance
            elif message.kind == "flush_timeout":
                if parts:
                    utterance = " ".join(parts)
                    parts.clear()
                    logger.info("STT utterance (flush_timeout)", extra={"utterance": utterance})
                    yield utterance

        if parts:  # flush any trailing transcript on disconnect
            utterance = " ".join(parts)
            logger.info("STT utterance (disconnect flush)", extra={"utterance": utterance})
            yield utterance

    async def close(self) -> None:
        """Close the STT WebSocket gracefully (idempotent)."""
        ws, self._ws = self._ws, None
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "CloseStream"}))
                await ws.close()
        logger.debug("Deepgram STT connection closed")

    async def _keepalive(self) -> None:
        """Deepgram closes idle sockets; ping periodically with KeepAlive."""
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
            if self._ws is None:
                return
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "KeepAlive"}))


class DeepgramTTSClient:
    """Text-to-speech via Deepgram Aura (REST), returning raw μ-law bytes."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def speak(self, text: str) -> bytes:
        """Synthesise ``text`` into 8 kHz μ-law audio bytes."""
        text = (text or "").strip()
        if not text:
            return b""
        headers = {
            "Authorization": f"Token {self._settings.deepgram_api_key}",
            "Content-Type": "application/json",
        }
        logger.debug("Deepgram TTS request", extra={"chars": len(text)})
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self._settings.deepgram_tts_speak_url,
                    headers=headers,
                    json={"text": text},
                )
                response.raise_for_status()
                audio = response.content
                logger.info(
                    "Deepgram TTS synthesised",
                    extra={"chars": len(text), "bytes": len(audio)},
                )
                return audio
        except httpx.HTTPError as exc:
            raise VoiceClientError(f"Deepgram TTS request failed: {exc}") from exc


__all__ = [
    "DeepgramSTTClient",
    "DeepgramTTSClient",
    "STTMessage",
    "VoiceClientError",
    "parse_stt_message",
]
