"""Tests for the Deepgram STT/TTS clients and audio framing helpers.

No network: WebSocket transport is faked and TTS HTTP calls use an
``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx

import app.voice.deepgram_client as deepgram_module
from app.config import get_settings
from app.telephony.audio_utils import chunk_bytes
from app.voice.deepgram_client import (
    DeepgramSTTClient,
    DeepgramTTSClient,
    VoiceClientError,
    parse_stt_message,
)


# ── STT message parsing ──────────────────────────────────────────────────────


def test_parse_final_transcript():
    msg = parse_stt_message(
        {
            "type": "Results",
            "is_final": True,
            "speech_final": True,
            "channel": {"alternatives": [{"transcript": "Yes, speaking.", "confidence": 0.98}]},
        }
    )
    assert msg.kind == "transcript_final"
    assert msg.transcript == "Yes, speaking."
    assert msg.is_final and msg.speech_final


def test_parse_interim_transcript():
    msg = parse_stt_message(
        {
            "type": "Results",
            "is_final": False,
            "speech_final": False,
            "channel": {"alternatives": [{"transcript": "yes spe"}]},
        }
    )
    assert msg.kind == "transcript_interim"
    assert not msg.is_final


def test_parse_utterance_end_and_unknown():
    assert parse_stt_message({"type": "UtteranceEnd"}).kind == "utterance_end"
    assert parse_stt_message({"type": "Metadata"}).kind == "other"


# ── Fake WebSocket transport ─────────────────────────────────────────────────


class FakeWebSocket:
    def __init__(
        self,
        incoming: Optional[List[Dict[str, Any]]] = None,
        block_on_empty: bool = False,
    ) -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()
        for message in incoming or []:
            self.incoming.put_nowait(json.dumps(message))
        self.sent_text: List[Dict[str, Any]] = []
        self.sent_bytes: List[bytes] = []
        self.closed = False
        self.block_on_empty = block_on_empty

    async def send(self, data: Any) -> None:
        if isinstance(data, (bytes, bytearray)):
            self.sent_bytes.append(bytes(data))
        else:
            self.sent_text.append(json.loads(data))

    async def recv(self) -> str:
        try:
            return self.incoming.get_nowait()
        except asyncio.QueueEmpty:
            if self.block_on_empty:
                await asyncio.sleep(3600)  # Simulate an open, idle socket
            raise ConnectionError("websocket closed")

    async def close(self) -> None:
        self.closed = True


def _patch_connect(monkeypatch, fake_ws: FakeWebSocket) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    async def fake_connect(url: str, **kwargs: Any) -> FakeWebSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return fake_ws

    monkeypatch.setattr(deepgram_module, "_ws_connect", fake_connect)
    return captured


# ── STT client behaviour ─────────────────────────────────────────────────────


async def test_stt_connect_uses_token_auth_and_mulaw_params(monkeypatch):
    fake_ws = FakeWebSocket(block_on_empty=True)
    captured = _patch_connect(monkeypatch, fake_ws)

    client = DeepgramSTTClient(get_settings(), handshake_timeout=0.05, flush_timeout=0.05)
    await client.connect()

    assert captured["url"].startswith("wss://api.deepgram.com/v1/listen")
    assert "encoding=mulaw" in captured["url"]
    assert "sample_rate=8000" in captured["url"]
    headers = next(v for v in captured["kwargs"].values() if isinstance(v, dict))
    assert headers["Authorization"] == "Token dg-test-key"
    await client.close()


async def test_stt_sends_raw_audio_frames(monkeypatch):
    fake_ws = FakeWebSocket(block_on_empty=True)
    _patch_connect(monkeypatch, fake_ws)
    client = DeepgramSTTClient(get_settings(), handshake_timeout=0.05, flush_timeout=0.05)
    await client.connect()

    frame = b"\xff" * 160
    await client.send_audio(frame)

    assert fake_ws.sent_bytes == [frame]
    await client.close()


async def test_stt_utterances_accumulate_until_speech_final(monkeypatch):
    fake_ws = FakeWebSocket(
        incoming=[
            {"type": "Results", "is_final": True, "speech_final": False,
             "channel": {"alternatives": [{"transcript": "Can I"}]}},
            {"type": "Results", "is_final": True, "speech_final": True,
             "channel": {"alternatives": [{"transcript": "reschedule?"}]}},
            {"type": "Results", "is_final": True, "speech_final": False,
             "channel": {"alternatives": [{"transcript": "No thanks"}]}},
            {"type": "UtteranceEnd"},
        ]
    )
    _patch_connect(monkeypatch, fake_ws)
    client = DeepgramSTTClient(get_settings(), handshake_timeout=0.05, flush_timeout=0.05)
    await client.connect()

    utterances = [u async for u in client.utterances()]

    assert utterances == ["Can I reschedule?", "No thanks"]
    await client.close()


async def test_stt_close_is_idempotent(monkeypatch):
    fake_ws = FakeWebSocket(block_on_empty=True)
    _patch_connect(monkeypatch, fake_ws)
    client = DeepgramSTTClient(get_settings(), handshake_timeout=0.05, flush_timeout=0.05)
    await client.connect()

    await client.close()
    await client.close()

    assert fake_ws.sent_text[-1] == {"type": "CloseStream"}
    assert fake_ws.closed
    assert not client.is_connected


async def test_stt_handshake_error_frame_raises(monkeypatch):
    """Deepgram rejects bad credentials with an immediate Error frame."""
    fake_ws = FakeWebSocket(incoming=[{"type": "Error", "message": "Invalid API key"}])
    _patch_connect(monkeypatch, fake_ws)
    client = DeepgramSTTClient(get_settings(), handshake_timeout=0.05, flush_timeout=0.05)

    try:
        await client.connect()
        raise AssertionError("expected VoiceClientError")
    except VoiceClientError as exc:
        assert "Invalid API key" in str(exc)


async def test_stt_utterances_timeout_flush(monkeypatch):
    """Buffered transcript parts flush even when speech_final never arrives."""
    fake_ws = FakeWebSocket(
        incoming=[
            {"type": "Results", "is_final": False,
             "channel": {"alternatives": [{"transcript": "I need to"}]}},
            {"type": "Results", "is_final": True, "speech_final": False,
             "channel": {"alternatives": [{"transcript": "reschedule my appointment"}]}},
        ]
    )
    _patch_connect(monkeypatch, fake_ws)
    # block_on_empty → recv hangs: triggers the flush timeout after 0.05s of silence
    fake_ws.block_on_empty = True
    client = DeepgramSTTClient(get_settings(), handshake_timeout=0.05, flush_timeout=0.05)
    await client.connect()

    # Collect utterances manually: first yield must be the timeout-flushed parts.
    gen = client.utterances()
    utterance = await gen.__anext__()
    assert utterance == "reschedule my appointment"
    await client.close()
    try:
        await gen.__anext__()
        raise AssertionError("generator should terminate after close")
    except StopAsyncIteration:
        pass


async def test_stt_utterances_flush_on_disconnect(monkeypatch):
    """Trailing parts flush when the socket closes."""
    fake_ws = FakeWebSocket(
        incoming=[
            {
                "type": "Results",
                "is_final": True,
                "speech_final": False,
                "channel": {"alternatives": [{"transcript": "trailing transcript"}]},
            },
        ]
    )
    _patch_connect(monkeypatch, fake_ws)
    client = DeepgramSTTClient(get_settings(), handshake_timeout=0.05, flush_timeout=0.05)
    await client.connect()

    utterances = [u async for u in client.utterances()]

    assert utterances == ["trailing transcript"]
    await client.close()


# ── TTS client behaviour (httpx.MockTransport) ───────────────────────────────


async def test_tts_returns_mulaw_bytes(monkeypatch):
    expected_audio = b"\xaa" * 800

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Token dg-test-key"
        assert request.url.path == "/v1/speak"
        assert "encoding=mulaw" in str(request.url)
        body = json.loads(request.content)
        assert body["text"] == "Hello there."
        return httpx.Response(200, content=expected_audio)

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client_cls(transport=transport, **kw)
    )

    client = DeepgramTTSClient(get_settings())
    audio = await client.speak("Hello there.")

    assert audio == expected_audio


async def test_tts_error_raises_voice_client_error(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"err": "bad key"})

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client_cls(transport=transport, **kw)
    )

    client = DeepgramTTSClient(get_settings())
    try:
        await client.speak("boom")
        raise AssertionError("expected VoiceClientError")
    except VoiceClientError:
        pass


async def test_tts_empty_text_returns_empty_bytes():
    client = DeepgramTTSClient(get_settings())
    assert await client.speak("   ") == b""


# ── Framing helper ───────────────────────────────────────────────────────────


def test_chunk_bytes_frames():
    audio = b"\x01" * 500
    frames = chunk_bytes(audio)  # default 160-byte / 20 ms frames
    assert [len(f) for f in frames] == [160, 160, 160, 20]
