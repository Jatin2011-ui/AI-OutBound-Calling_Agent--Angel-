"""Audio conversion helpers for the telephony pipeline.

Twilio Media Streams transport G.711 μ-law, 8 kHz, 8-bit, mono audio.
Deepgram accepts/emits the same ``mulaw``/8 kHz format, so in the
hot path audio is passed through unchanged as base64. These utilities exist
for resampling/analysis use cases (recording, testing, alternate providers).

``audioop`` is stdlib on Python ≤3.12 and provided by the ``audioop-lts``
backport on Python 3.13+.
"""

from __future__ import annotations

import base64

try:
    import audioop  # stdlib on Python ≤ 3.12
except ImportError:  # Python 3.13+ — stdlib audioop was removed
    import audioop_lts as audioop  # type: ignore[no-redef]  # backport

PCM_SAMPLE_WIDTH = 2  # 16-bit PCM


class AudioConversionError(Exception):
    """Raised when an audio conversion fails."""


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Convert G.711 μ-law (8-bit) bytes to 16-bit linear PCM bytes."""
    try:
        return audioop.ulaw2lin(mulaw_bytes, PCM_SAMPLE_WIDTH)
    except audioop.error as exc:
        raise AudioConversionError(f"μ-law → PCM conversion failed: {exc}") from exc


def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Convert 16-bit linear PCM bytes to G.711 μ-law (8-bit) bytes."""
    try:
        return audioop.lin2ulaw(pcm_bytes, PCM_SAMPLE_WIDTH)
    except audioop.error as exc:
        raise AudioConversionError(f"PCM → μ-law conversion failed: {exc}") from exc


def resample_pcm16(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample 16-bit mono PCM between sample rates (stateless helper)."""
    if from_rate == to_rate:
        return pcm_bytes
    try:
        converted, _ = audioop.ratecv(pcm_bytes, PCM_SAMPLE_WIDTH, 1, from_rate, to_rate, None)
        return converted
    except audioop.error as exc:
        raise AudioConversionError(f"PCM resample failed: {exc}") from exc


TWILIO_FRAME_BYTES = 160  # 20 ms of 8 kHz 8-bit μ-law mono audio


def chunk_bytes(payload: bytes, size: int = TWILIO_FRAME_BYTES) -> list[bytes]:
    """Split raw audio into fixed-size frames for Twilio media messages."""
    return [payload[i : i + size] for i in range(0, len(payload), size)]


def bytes_to_base64(payload: bytes) -> str:
    """Encode raw audio bytes to the base64 string both APIs expect."""
    return base64.b64encode(payload).decode("ascii")


def base64_to_bytes(payload: str) -> bytes:
    """Decode a base64 audio chunk back to raw bytes."""
    return base64.b64decode(payload.encode("ascii"))


__all__ = [
    "AudioConversionError",
    "mulaw_to_pcm16",
    "pcm16_to_mulaw",
    "resample_pcm16",
    "bytes_to_base64",
    "base64_to_bytes",
    "chunk_bytes",
    "PCM_SAMPLE_WIDTH",
    "TWILIO_FRAME_BYTES",
]
