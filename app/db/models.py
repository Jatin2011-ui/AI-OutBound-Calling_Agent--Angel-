"""SQLAlchemy ORM models.

Tables mirror ``app/db/migrations/001_initial.sql``:

- :class:`Call` — one row per outbound call.
- :class:`Transcript` — per-turn conversation lines for a call.
- :class:`KnowledgeBase` — semantic memory documents with pgvector embeddings.

Note: ``meta`` attributes map to the ``metadata`` columns (the bare name
``metadata`` is reserved by SQLAlchemy's declarative API).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Call(Base):
    """An outbound call record."""

    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    call_sid: Mapped[str] = mapped_column(String, unique=True, index=True)
    to_number: Mapped[str] = mapped_column(String)
    from_number: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="initiated")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    duration_s: Mapped[Optional[int]] = mapped_column(nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, server_default="{}")

    transcripts: Mapped[List["Transcript"]] = relationship(
        back_populates="call", cascade="all, delete-orphan", order_by="Transcript.turn_number"
    )


class Transcript(Base):
    """A single conversation turn within a call."""

    __tablename__ = "transcripts"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String)  # 'agent' | 'caller'
    content: Mapped[str] = mapped_column(Text)
    turn_number: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    call: Mapped[Call] = relationship(back_populates="transcripts")


class KnowledgeBase(Base):
    """A semantic memory / knowledge document with an embedding vector."""

    __tablename__ = "knowledge_base"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    content: Mapped[str] = mapped_column(Text)
    # 384 dimensions == all-MiniLM-L6-v2 output. Keep in sync with
    # app/db/migrations/001_initial.sql and settings.pgvector_dimension.
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


__all__ = ["Base", "Call", "Transcript", "KnowledgeBase"]
