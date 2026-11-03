"""FastAPI dependency injection and process-wide service registry.

Owns singletons (settings, DB engine/session factory, asyncpg pool, Twilio
client, memory store, response generator, call persister) used across the
app, plus an ``overrides`` registry so tests can substitute fakes without
monkey-patching module attributes::

    dependencies.set_override("memory_store", FakeMemoryStore())
    ...
    dependencies.clear_overrides()

``initialise``/``shutdown`` are driven by the FastAPI lifespan in
:mod:`app.main`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Protocol, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import asyncpg

from app.agent.state import Message
from app.config import Settings, get_settings as _load_settings
from app.db import connection as db_connection
from app.db.models import Call, Transcript
from app.memory.embeddings import EmbeddingClient
from app.memory.store import MemoryStore
from app.telephony.twilio_client import TwilioCallClient

logger = logging.getLogger(__name__)


# ── Protocols for injectable services ────────────────────────────────────────


class ResponseGenerator(Protocol):
    """Produces the agent's next utterance text."""

    async def __call__(self, *, system_prompt: str, history: List[Message]) -> str: ...


class CallPersister(Protocol):
    """Persists call lifecycle information."""

    async def mark_escalated(self, call_sid: str) -> None: ...

    async def complete_call(
        self, call_sid: str, summary: str, history: List[Message]
    ) -> None: ...


class CallRepository(Protocol):
    """Read/write access to call records for the HTTP API."""

    async def record_initiated(
        self, call_sid: str, to_number: str, from_number: str, context: Dict[str, Any]
    ) -> str: ...

    async def get_call(self, call_id: str) -> Optional[Dict[str, Any]]: ...

    async def list_calls(self) -> List[Dict[str, Any]]: ...

    async def update_status(self, call_sid: str, status: str) -> None: ...


IntentClassifier = Callable[[str], Tuple[str, str]]


# ── Default implementations ──────────────────────────────────────────────────


class GroqChatResponder:
    """Default responder backed by Groq chat completions (e.g. llama-3.3-70b).

    Uses the official ``groq`` SDK, which speaks to Groq's OpenAI-compatible
    chat completions endpoint.
    """

    def __init__(self, settings: Settings) -> None:
        from groq import AsyncGroq

        self._client = AsyncGroq(
            api_key=settings.groq_api_key or "gsk-unconfigured",
            base_url=settings.groq_base_url,
        )
        self._model = settings.groq_chat_model

    async def __call__(self, *, system_prompt: str, history: List[Message]) -> str:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for message in history[-20:]:  # keep prompts bounded
            role = "assistant" if message["role"] == "agent" else "user"
            messages.append({"role": role, "content": message["content"]})
        response = await self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=200, temperature=0.4
        )
        return (response.choices[0].message.content or "").strip()


class SqlCallPersister:
    """Persists call state transitions to Postgres via SQLAlchemy."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def mark_escalated(self, call_sid: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(Call).where(Call.call_sid == call_sid).values(status="escalated")
            )
            await session.commit()

    async def complete_call(
        self, call_sid: str, summary: str, history: List[Message]
    ) -> None:
        async with self._session_factory() as session:
            result = await session.execute(select(Call).where(Call.call_sid == call_sid))
            call = result.scalar_one_or_none()
            ended_at = datetime.now(timezone.utc)
            if call is not None:
                duration: Optional[int] = None
                if call.started_at is not None:
                    started_at = call.started_at
                    if started_at.tzinfo is None:
                        started_at = started_at.replace(tzinfo=timezone.utc)
                    duration = max(0, int((ended_at - started_at).total_seconds()))
                await session.execute(
                    update(Call)
                    .where(Call.id == call.id)
                    .values(status="completed", summary=summary, ended_at=ended_at,
                            duration_s=duration)
                )
                for index, message in enumerate(history):
                    session.add(
                        Transcript(
                            call_id=call.id,
                            role=message["role"],
                            content=message["content"],
                            turn_number=index,
                        )
                    )
                await session.commit()
            else:
                logger.warning(
                    "complete_call: no call row found", extra={"call_sid": call_sid}
                )


class NullCallPersister:
    """No-op persister used when no database is configured."""

    async def mark_escalated(self, call_sid: str) -> None:
        logger.debug("NullCallPersister.mark_escalated", extra={"call_sid": call_sid})

    async def complete_call(
        self, call_sid: str, summary: str, history: List[Message]
    ) -> None:
        logger.debug(
            "NullCallPersister.complete_call",
            extra={"call_sid": call_sid, "turns": len(history)},
        )


class SqlCallRepository:
    """SQL-backed call record repository for the REST API."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record_initiated(
        self, call_sid: str, to_number: str, from_number: str, context: Dict[str, Any]
    ) -> str:
        async with self._session_factory() as session:
            call = Call(
                call_sid=call_sid,
                to_number=to_number,
                from_number=from_number,
                status="initiated",
                meta={"context": context},
            )
            session.add(call)
            await session.commit()
            return str(call.id)

    async def get_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Call).where(Call.id == uuid.UUID(call_id))
            )
            call = result.scalar_one_or_none()
            if call is None:
                return None
            return {
                "call_id": str(call.id),
                "call_sid": call.call_sid,
                "to_number": call.to_number,
                "from_number": call.from_number,
                "status": call.status,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                "duration_s": call.duration_s,
                "summary": call.summary,
                "metadata": call.meta,
                "transcript": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "turn_number": t.turn_number,
                    }
                    for t in call.transcripts
                ],
            }

    async def list_calls(self) -> List[Dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Call).order_by(Call.started_at.desc())
            )
            calls = result.scalars().all()
            return [
                {
                    "call_id": str(c.id),
                    "call_sid": c.call_sid,
                    "to_number": c.to_number,
                    "from_number": c.from_number,
                    "status": c.status,
                    "started_at": c.started_at.isoformat() if c.started_at else None,
                    "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                    "duration_s": c.duration_s,
                    "summary": c.summary,
                    "context": c.meta.get("context", {}) if c.meta else {},
                }
                for c in calls
            ]

    async def update_status(self, call_sid: str, status: str) -> None:
        terminal = status in {"completed", "canceled", "busy", "failed", "no-answer"}
        async with self._session_factory() as session:
            values: Dict[str, Any] = {"status": status}
            if terminal:
                result = await session.execute(
                    select(Call).where(Call.call_sid == call_sid)
                )
                call = result.scalar_one_or_none()
                if call is not None and call.ended_at is None:
                    ended_at = datetime.now(timezone.utc)
                    values["ended_at"] = ended_at
                    if call.started_at is not None:
                        started_at = call.started_at
                        if started_at.tzinfo is None:
                            started_at = started_at.replace(tzinfo=timezone.utc)
                        values["duration_s"] = max(
                            0, int((ended_at - started_at).total_seconds())
                        )
            await session.execute(
                update(Call).where(Call.call_sid == call_sid).values(**values)
            )
            await session.commit()


_TERMINAL_STATUSES = frozenset({"completed", "canceled", "busy", "failed", "no-answer"})


class InMemoryCallRepository:
    """Fallback repository when the database is unavailable (dev/test).

    Stores call records, status transitions, AND closing artefacts (summary +
    transcript) in process memory so ``GET /calls/{id}`` still returns a
    complete record when Postgres is down.
    """

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}
        self._by_sid: Dict[str, str] = {}

    async def record_initiated(
        self, call_sid: str, to_number: str, from_number: str, context: Dict[str, Any]
    ) -> str:
        call_id = str(uuid.uuid4())
        self._records[call_id] = {
            "call_id": call_id,
            "call_sid": call_sid,
            "to_number": to_number,
            "from_number": from_number,
            "status": "initiated",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "duration_s": None,
            "summary": None,
            "metadata": {"context": context},
            "transcript": [],
        }
        self._by_sid[call_sid] = call_id
        return call_id

    async def get_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        return self._records.get(call_id)

    async def list_calls(self) -> List[Dict[str, Any]]:
        """Return all call records sorted newest-first, with context extracted."""
        rows = list(self._records.values())
        rows.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return [
            {
                "call_id": r["call_id"],
                "call_sid": r["call_sid"],
                "to_number": r["to_number"],
                "from_number": r["from_number"],
                "status": r["status"],
                "started_at": r.get("started_at"),
                "ended_at": r.get("ended_at"),
                "duration_s": r.get("duration_s"),
                "summary": r.get("summary"),
                "context": r.get("metadata", {}).get("context", {}),
            }
            for r in rows
        ]

    async def update_status(self, call_sid: str, status: str) -> None:
        record = self._record_by_sid(call_sid)
        if record is not None:
            record["status"] = status
            if status in _TERMINAL_STATUSES and record["ended_at"] is None:
                self._close_out_record(record)

    def record_completion(
        self, call_sid: str, summary: str, history: List[Message]
    ) -> None:
        """Persist closing artefacts (idempotent per call)."""
        record = self._record_by_sid(call_sid)
        if record is None:
            logger.warning(
                "record_completion: unknown call", extra={"call_sid": call_sid}
            )
            return
        record["status"] = "completed"
        record["summary"] = summary
        record["transcript"] = [
            {"role": m["role"], "content": m["content"], "turn_number": i}
            for i, m in enumerate(history)
        ]
        if record["ended_at"] is None:
            self._close_out_record(record)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _record_by_sid(self, call_sid: str) -> Optional[Dict[str, Any]]:
        call_id = self._by_sid.get(call_sid)
        return self._records.get(call_id) if call_id else None

    @staticmethod
    def _close_out_record(record: Dict[str, Any]) -> None:
        ended_at = datetime.now(timezone.utc)
        record["ended_at"] = ended_at.isoformat()
        started_raw = record.get("started_at")
        if started_raw:
            started_at = datetime.fromisoformat(started_raw)
            record["duration_s"] = max(0, int((ended_at - started_at).total_seconds()))


class InMemoryCallPersister:
    """Persister used when Postgres is unavailable: writes into the
    :class:`InMemoryCallRepository` so call outcomes remain queryable."""

    def __init__(self, repository: InMemoryCallRepository) -> None:
        self._repository = repository

    async def mark_escalated(self, call_sid: str) -> None:
        await self._repository.update_status(call_sid, "escalated")

    async def complete_call(
        self, call_sid: str, summary: str, history: List[Message]
    ) -> None:
        self._repository.record_completion(call_sid, summary, history)


# ── Registry state ───────────────────────────────────────────────────────────


class _Registry:
    engine: Optional[AsyncEngine] = None
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    pg_pool: Optional[asyncpg.Pool] = None
    memory_store: Optional[MemoryStore] = None
    embedding_client: Optional[EmbeddingClient] = None
    twilio_client: Optional[TwilioCallClient] = None
    responder: Optional[ResponseGenerator] = None
    call_repository: Optional[CallRepository] = None


_registry = _Registry()
_overrides: Dict[str, Any] = {}


def set_override(name: str, value: Any) -> None:
    """Register a test/live-call override for a dependency by name."""
    _overrides[name] = value


def clear_overrides() -> None:
    """Remove all registered overrides (called between tests)."""
    _overrides.clear()


# ── Lifecycle ────────────────────────────────────────────────────────────────


async def initialise(settings: Settings) -> None:
    """Initialise database-backed services. Failures degrade to in-memory."""
    try:
        _registry.engine = db_connection.create_engine(settings)
        _registry.session_factory = db_connection.create_session_factory(_registry.engine)
        healthy = await db_connection.check_database_health(_registry.engine)
        if not healthy:
            raise ConnectionError("database health check failed")
    except Exception as exc:  # noqa: BLE001 - app still serves without DB
        logger.error("Database initialisation failed; running degraded", extra={"error": str(exc)})
        _registry.engine = None
        _registry.session_factory = None

    try:
        _registry.pg_pool = await asyncpg.create_pool(
            dsn=settings.asyncpg_dsn, min_size=1, max_size=10
        )
    except Exception as exc:  # noqa: BLE001 - memory features degrade only
        logger.error("pgvector pool creation failed; memory disabled", extra={"error": str(exc)})
        _registry.pg_pool = None

    logger.info(
        "Dependencies initialised",
        extra={
            "db": _registry.engine is not None,
            "pgvector": _registry.pg_pool is not None,
        },
    )


async def shutdown() -> None:
    """Tear down all managed services."""
    if _registry.pg_pool is not None:
        try:
            await _registry.pg_pool.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing pg pool", extra={"error": str(exc)})
        _registry.pg_pool = None
    if _registry.engine is not None:
        await db_connection.dispose_engine(_registry.engine)
        _registry.engine = None
        _registry.session_factory = None
    _registry.memory_store = None
    _registry.call_repository = None


# ── Getters (override-aware) ─────────────────────────────────────────────────


def get_settings() -> Settings:
    if "settings" in _overrides:
        return _overrides["settings"]
    return _load_settings()


def get_engine() -> AsyncEngine:
    if "engine" in _overrides:
        return _overrides["engine"]
    if _registry.engine is None:
        raise RuntimeError("Database engine not initialised")
    return _registry.engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if "session_factory" in _overrides:
        return _overrides["session_factory"]
    if _registry.session_factory is None:
        raise RuntimeError("Session factory not initialised")
    return _registry.session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a database session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


def get_embedding_client() -> EmbeddingClient:
    if "embedding_client" in _overrides:
        return _overrides["embedding_client"]
    if _registry.embedding_client is None:
        settings = get_settings()
        _registry.embedding_client = EmbeddingClient(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
    return _registry.embedding_client


def get_memory_store() -> MemoryStore:
    if "memory_store" in _overrides:
        return _overrides["memory_store"]
    if _registry.memory_store is None:
        if _registry.pg_pool is None:
            raise RuntimeError("pgvector pool not initialised")
        settings = get_settings()
        _registry.memory_store = MemoryStore(
            pool=_registry.pg_pool,
            embedder=get_embedding_client(),
            default_top_k=settings.memory_retrieval_top_k,
        )
    return _registry.memory_store


def get_twilio_client() -> TwilioCallClient:
    if "twilio_client" in _overrides:
        return _overrides["twilio_client"]
    if _registry.twilio_client is None:
        settings = get_settings()
        _registry.twilio_client = TwilioCallClient(
            account_sid=settings.twilio_account_sid,
            auth_token=settings.twilio_auth_token,
            from_number=settings.twilio_phone_number,
        )
    return _registry.twilio_client


def get_response_generator() -> ResponseGenerator:
    if "response_generator" in _overrides:
        return _overrides["response_generator"]
    if _registry.responder is None:
        _registry.responder = GroqChatResponder(get_settings())
    return _registry.responder


def get_intent_classifier() -> IntentClassifier:
    if "intent_classifier" in _overrides:
        return _overrides["intent_classifier"]
    # Imported lazily to avoid an import cycle with the agent nodes.
    from app.agent.nodes.query_handler import heuristic_classify

    return heuristic_classify


def get_call_persister() -> CallPersister:
    """Closing/escalation writer.

    With a database: :class:`SqlCallPersister`. Degraded (no DB): an
    :class:`InMemoryCallPersister` backed by the shared in-memory repository
    so summaries/transcripts still show up in ``GET /calls/{id}``.
    """
    if "call_persister" in _overrides:
        return _overrides["call_persister"]
    if _registry.session_factory is not None:
        return SqlCallPersister(_registry.session_factory)
    repository = get_call_repository()
    if isinstance(repository, InMemoryCallRepository):
        return InMemoryCallPersister(repository)
    return NullCallPersister()


def get_call_repository() -> CallRepository:
    if "call_repository" in _overrides:
        return _overrides["call_repository"]
    if _registry.call_repository is None:
        if _registry.session_factory is not None:
            _registry.call_repository = SqlCallRepository(_registry.session_factory)
        else:
            _registry.call_repository = InMemoryCallRepository()
    return _registry.call_repository


__all__ = [
    "initialise",
    "shutdown",
    "set_override",
    "clear_overrides",
    "get_settings",
    "get_engine",
    "get_session_factory",
    "get_db_session",
    "get_embedding_client",
    "get_memory_store",
    "get_twilio_client",
    "get_response_generator",
    "get_intent_classifier",
    "get_call_persister",
    "get_call_repository",
    "ResponseGenerator",
    "CallPersister",
    "CallRepository",
    "IntentClassifier",
    "GroqChatResponder",
    "SqlCallPersister",
    "NullCallPersister",
    "SqlCallRepository",
    "InMemoryCallRepository",
    "InMemoryCallPersister",
]
