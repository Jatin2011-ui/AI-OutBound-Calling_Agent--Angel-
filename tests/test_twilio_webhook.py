"""Tests for the Twilio webhook endpoints and TwiML generation.

Uses FastAPI's TestClient; with ``APP_ENV=test`` the lifespan skips DB setup
and Twilio signature validation is bypassed (enforced only in production).
"""

from __future__ import annotations

import json
import uuid

from fastapi.testclient import TestClient

from app.main import create_app
from app.telephony.twiml_builder import build_say_twiml, build_stream_twiml


def _client() -> TestClient:
    return TestClient(create_app())


# ── TwiML builders ───────────────────────────────────────────────────────────


def test_build_stream_twiml_structure():
    twiml = build_stream_twiml("wss://example.com/ws/media-stream/CA123")
    assert twiml.startswith("<?xml")
    assert "<Response>" in twiml
    assert "<Connect>" in twiml
    assert '<Stream url="wss://example.com/ws/media-stream/CA123"' in twiml


def test_build_say_twiml_escapes_content():
    twiml = build_say_twiml("Fish & Chips <special>")
    assert "Fish &amp; Chips &lt;special&gt;" in twiml
    assert "<Hangup />" in twiml


# ── Webhook routes ───────────────────────────────────────────────────────────


def test_twiml_webhook_returns_stream_url():
    with _client() as client:
        response = client.get("/webhooks/twilio/twiml", params={"CallSid": "CA_TEST_999"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "<Stream" in response.text
    assert "/ws/media-stream/CA_TEST_999" in response.text


def test_twiml_webhook_accepts_post():
    with _client() as client:
        response = client.post(
            "/webhooks/twilio/twiml",
            data={"CallSid": "CA_POST_1", "From": "+15550101010"},
        )
    assert response.status_code == 200
    assert "CA_POST_1" in response.text


def test_twiml_webhook_forwards_context_to_stream_url():
    context_json = json.dumps({"customer_name": "Jatin", "purpose": "appointment reminder"})
    with _client() as client:
        response = client.get(
            "/webhooks/twilio/twiml",
            params={"CallSid": "CA_CTX_1", "context": context_json},
        )
    assert response.status_code == 200
    assert "/ws/media-stream/CA_CTX_1" in response.text
    assert "context=" in response.text


def test_status_callback_is_acknowledged():
    with _client() as client:
        response = client.post(
            "/webhooks/twilio/status",
            data={"CallSid": "CA_TEST_123", "CallStatus": "completed"},
        )
    assert response.status_code == 204


# ── Calls API ────────────────────────────────────────────────────────────────


def test_initiate_call_uses_twilio_and_records(fake_twilio):
    with _client() as client:
        response = client.post(
            "/calls/initiate",
            json={
                "to_number": "+15557654321",
                "context": {"customer_name": "Priya Sharma", "purpose": "appointment_reminder"},
            },
        )
    assert response.status_code == 201
    payload = response.json()
    assert payload["call_sid"] == "CA_TEST_123"
    assert payload["to_number"] == "+15557654321"
    assert payload["status"] == "queued"
    # Twilio was pointed back at our webhook endpoints, with the call context
    # URL-encoded onto the TwiML URL for forwarding into the WS.
    twiml_url = fake_twilio.initiated[0]["twiml_url"]
    assert twiml_url.startswith("https://test.ngrok.io/webhooks/twilio/twiml")
    assert "context=" in twiml_url
    assert "Priya" in twiml_url
    assert "/webhooks/twilio/status" in fake_twilio.initiated[0]["status_callback_url"]


def test_initiate_call_rejects_bad_number():
    with _client() as client:
        response = client.post("/calls/initiate", json={"to_number": "not-a-number"})
    assert response.status_code == 422


def test_get_unknown_call_returns_404():
    with _client() as client:
        response = client.get(f"/calls/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_in_memory_fallback_persists_call_artefacts():
    """With Postgres down, closing artefacts still land on the in-memory record."""
    from app.agent.state import Message, utcnow
    from app.api import dependencies

    repository = dependencies.InMemoryCallRepository()
    persister = dependencies.InMemoryCallPersister(repository)

    call_id = await repository.record_initiated(
        "CA_TEST_INMEM", "+15557654321", "+15551234567", {"customer_name": "Jatin"}
    )
    await persister.complete_call(
        "CA_TEST_INMEM",
        "The caller confirmed the appointment.",
        [Message(role="agent", content="great!", timestamp=utcnow()),
         Message(role="caller", content="sounds good", timestamp=utcnow())],
    )

    record = await repository.get_call(call_id)

    assert record["summary"] == "The caller confirmed the appointment."
    assert record["status"] == "completed"
    assert record["ended_at"] is not None
    assert record["duration_s"] is not None
    assert record["transcript"][0]["role"] == "agent"
    assert record["transcript"][1]["role"] == "caller"



def test_health_endpoint():
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
