"""Shared pytest fixtures and fakes.

Environment variables are forced to test-safe values *before* any ``app``
module is imported, so Settings pick them up and external services are never
touched.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["APP_ENV"] = "test"
os.environ.setdefault("GROQ_API_KEY", "gsk-test")
os.environ.setdefault("DEEPGRAM_API_KEY", "dg-test-key")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "ACtestaccountsid")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "testauthtoken")
os.environ.setdefault("TWILIO_PHONE_NUMBER", "+15551234567")
os.environ.setdefault("TWILIO_WEBHOOK_BASE_URL", "https://test.ngrok.io")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.agent.state import AgentState, Message, new_agent_state, utcnow  # noqa: E402
from app.api import dependencies  # noqa: E402
from app.memory.schemas import MemoryRecordCreate, MemorySearchResult  # noqa: E402
from app.telephony.twilio_client import CallHandle  # noqa: E402


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeMemoryStore:
    """In-memory replacement for the pgvector store."""

    def __init__(self, results: Optional[List[MemorySearchResult]] = None) -> None:
        self.results = list(results or [])
        self.searches: List[str] = []
        self.upserted: List[MemoryRecordCreate] = []

    async def search(self, query_text: str, top_k: Optional[int] = None):
        self.searches.append(query_text)
        return list(self.results)

    async def upsert(self, record: MemoryRecordCreate):
        self.upserted.append(record)
        return uuid.uuid4()


class FakeResponder:
    """Scripted text responder for response_gen / closing nodes."""

    def __init__(self, replies: Optional[List[str]] = None) -> None:
        self.replies = list(replies or [])
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, *, system_prompt: str, history: List[Message]) -> str:
        self.calls.append({"system_prompt": system_prompt, "history": history})
        # Return "" when no replies are scripted so nodes fall back to their
        # static/deterministic fallback paths (e.g. greeting_node static greeting).
        return self.replies.pop(0) if self.replies else ""


class RecordingCallPersister:
    """Records persistence calls instead of writing to a database."""

    def __init__(self) -> None:
        self.escalated: List[str] = []
        self.completed: List[Dict[str, Any]] = []

    async def mark_escalated(self, call_sid: str) -> None:
        self.escalated.append(call_sid)

    async def complete_call(self, call_sid: str, summary: str, history: List[Message]) -> None:
        self.completed.append({"call_sid": call_sid, "summary": summary, "history": history})


class FakeTwilioClient:
    """Returns canned call handles without touching the Twilio API."""

    def __init__(self) -> None:
        self.initiated: List[Dict[str, Any]] = []
        self.ended: List[str] = []

    async def initiate_call(self, to_number, twiml_url=None, status_callback_url=None):
        self.initiated.append(
            {
                "to_number": to_number,
                "twiml_url": twiml_url,
                "status_callback_url": status_callback_url,
            }
        )
        return CallHandle(
            call_sid="CA_TEST_123",
            status="queued",
            to_number=to_number,
            from_number="+15551234567",
        )

    async def end_call(self, call_sid: str) -> bool:
        self.ended.append(call_sid)
        return True

    async def fetch_call_status(self, call_sid: str) -> Optional[str]:
        return "completed"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_overrides():
    """Guarantee a clean dependency-override registry per test."""
    dependencies.clear_overrides()
    yield
    dependencies.clear_overrides()


@pytest.fixture
def base_state() -> AgentState:
    """Fresh agent state for a synthetic call."""
    return new_agent_state(
        call_sid="CA_TEST_123",
        metadata={"context": {"customer_name": "Priya Sharma", "purpose": "appointment_reminder"}},
    )


@pytest.fixture
def fake_responder() -> FakeResponder:
    responder = FakeResponder()
    dependencies.set_override("response_generator", responder)
    return responder


@pytest.fixture
def fake_memory_store() -> FakeMemoryStore:
    store = FakeMemoryStore()
    dependencies.set_override("memory_store", store)
    return store


@pytest.fixture
def fake_persister() -> RecordingCallPersister:
    persister = RecordingCallPersister()
    dependencies.set_override("call_persister", persister)
    return persister


@pytest.fixture
def fake_twilio() -> FakeTwilioClient:
    twilio = FakeTwilioClient()
    dependencies.set_override("twilio_client", twilio)
    return twilio


def caller_message(content: str) -> Message:
    return Message(role="caller", content=content, timestamp=utcnow())


def agent_message(content: str) -> Message:
    return Message(role="agent", content=content, timestamp=utcnow())
