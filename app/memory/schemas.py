"""Pydantic models for memory records flowing in/out of the pgvector store."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryRecordCreate(BaseModel):
    """Payload required to insert a new memory document."""

    content: str = Field(min_length=1)
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    """A fully materialised memory record as stored in Postgres."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    content: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None


class MemorySearchResult(BaseModel):
    """A single hit from a cosine-similarity search."""

    id: uuid.UUID
    content: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: float = Field(description="Cosine distance returned by `<=>` (0 == identical).")


__all__ = ["MemoryRecordCreate", "MemoryRecord", "MemorySearchResult"]
