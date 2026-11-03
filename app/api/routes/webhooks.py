"""Twilio webhook endpoints.

- ``GET/POST /webhooks/twilio/twiml`` — served when the outbound call is
  answered; returns TwiML that connects a Media Stream back to us.
- ``POST /webhooks/twilio/status`` — call lifecycle status callbacks.
- ``WebSocket /ws/media-stream/{call_sid}`` — the live audio bridge.

Twilio request signatures are validated with ``twilio.request_validator``
whenever the app runs in production.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, status
from twilio.request_validator import RequestValidator

from app.api import dependencies
from app.telephony.media_stream import MediaStreamHandler
from app.telephony.twiml_builder import build_stream_twiml

logger = logging.getLogger(__name__)

http_router = APIRouter(prefix="/webhooks/twilio", tags=["webhooks"])
ws_router = APIRouter(tags=["webhooks"])


def _validate_twilio_signature(
    url: str, params: Dict[str, str], signature: Optional[str]
) -> None:
    """Raise 403 unless the request carries a valid Twilio signature.

    Validation is enforced only in production; in development/test the
    signature header is usually absent (curl, TestClient).
    """
    settings = dependencies.get_settings()
    if not settings.is_production:
        return
    validator = RequestValidator(settings.twilio_auth_token)
    if not signature or not validator.validate(url, params, signature):
        logger.warning("Twilio signature validation failed", extra={"url": url})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Twilio signature"
        )


@http_router.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request) -> Response:
    """Return TwiML connecting the answered call to our Media Stream socket."""
    params: Dict[str, str] = {}
    if request.method == "POST":
        form = await request.form()
        params = {key: str(value) for key, value in form.items()}
    else:
        params = dict(request.query_params)

    _validate_twilio_signature(
        str(request.url), params, request.headers.get("X-Twilio-Signature")
    )

    call_sid = params.get("CallSid") or "unknown"
    settings = dependencies.get_settings()
    # Forward the call context (attached by /calls/initiate to this TwiML
    # URL) onto the Media Stream WebSocket URL so the WS handler receives it.
    stream_url = settings.media_stream_ws_url(
        call_sid, context_json=params.get("context")
    )
    logger.info(
        "Serving stream TwiML", extra={"call_sid": call_sid, "stream_url": stream_url}
    )
    return Response(
        content=build_stream_twiml(stream_url),
        media_type="application/xml",
    )


@http_router.post("/status", status_code=status.HTTP_204_NO_CONTENT)
async def status_callback(request: Request) -> Response:
    """Receive Twilio call status callbacks and update the call record."""
    form = await request.form()
    params = {key: str(value) for key, value in form.items()}
    _validate_twilio_signature(
        str(request.url), params, request.headers.get("X-Twilio-Signature")
    )

    call_sid = params.get("CallSid", "unknown")
    call_status = params.get("CallStatus", "unknown")
    logger.info(
        "Twilio status callback",
        extra={"call_sid": call_sid, "twilio_status": call_status},
    )
    try:
        repository = dependencies.get_call_repository()
        await repository.update_status(call_sid, call_status)
    except Exception as exc:  # noqa: BLE001 - ack regardless; never 5xx Twilio
        logger.warning(
            "Failed to record status callback",
            extra={"call_sid": call_sid, "error": str(exc)},
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@ws_router.websocket("/ws/media-stream/{call_sid}")
async def media_stream_ws(websocket: WebSocket, call_sid: str) -> None:
    """Bidirectional audio bridge: Twilio ⇄ Deepgram STT/TTS ⇄ Groq LLM."""
    await websocket.accept()
    raw_context = websocket.query_params.get("context")
    context: dict | None = None
    if raw_context:
        try:
            parsed = json.loads(raw_context)
            context = parsed if isinstance(parsed, dict) else {"raw": parsed}
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse context query param", extra={"call_sid": call_sid}
            )
            context = {"raw": raw_context}
    logger.info(
        "Media stream WebSocket accepted",
        extra={"call_sid": call_sid, "context_keys": sorted(context.keys()) if context else []},
    )
    handler = MediaStreamHandler(
        websocket=websocket,
        call_sid=call_sid,
        settings=dependencies.get_settings(),
        context=context,
    )
    try:
        await handler.run()
    except Exception as exc:  # noqa: BLE001 - log, then let the socket close
        logger.error(
            "Media stream handler crashed",
            extra={"call_sid": call_sid, "error": str(exc)},
        )


__all__ = ["http_router", "ws_router"]
