"""pgvector-backed memory store.

Uses a raw :mod:`asyncpg` connection pool because SQLAlchemy has no native
support for the ``vector`` type or the ``<=>`` cosine-distance operator.
Embeddings are passed as text literals and cast with ``::vector``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional, Sequence

import asyncpg

from app.memory.embeddings import EmbeddingClient
from app.memory.schemas import MemoryRecordCreate, MemorySearchResult

logger = logging.getLogger(__name__)


class MemoryStoreError(Exception):
    """Raised when a memory store operation fails against Postgres."""


def vector_to_literal(vector: Sequence[float]) -> str:
    """Serialise a float vector into the ``[v1,v2,...]`` literal pgvector expects."""
    return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"


class MemoryStore:
    """Read/write access to the ``knowledge_base`` table."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        embedder: Optional[EmbeddingClient] = None,
        default_top_k: int = 5,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self.default_top_k = default_top_k

    # ── Writes ──────────────────────────────────────────────────────────

    async def upsert(self, record: MemoryRecordCreate) -> Optional[uuid.UUID]:
        """Embed ``record.content`` and insert it into ``knowledge_base``.

        Returns the new row's UUID, or ``None`` if embedding generation failed
        (the caller can still proceed; the call must not crash on memory
        write failures).
        """
        embedding: List[float] = []
        if self._embedder is not None:
            try:
                embedding = await self._embedder.embed(record.content)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.error("Embedding generation failed during upsert", extra={"error": str(exc)})
        if not embedding:
            logger.warning("Skipping upsert: empty embedding", extra={"source": record.source})
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO knowledge_base (content, embedding, source, metadata)
                    VALUES ($1, $2::vector, $3, $4::jsonb)
                    RETURNING id
                    """,
                    record.content,
                    vector_to_literal(embedding),
                    record.source,
                    json.dumps(record.metadata),
                )
            return row["id"] if row else None
        except asyncpg.PostgresError as exc:
            raise MemoryStoreError(f"Failed to upsert memory record: {exc}") from exc

    # ── Reads ───────────────────────────────────────────────────────────

    async def search(self, query_text: str, top_k: Optional[int] = None) -> List[MemorySearchResult]:
        """Embed ``query_text`` and run cosine-similarity search."""
        if self._embedder is None:
            logger.warning("MemoryStore.search called without an embedder")
            return []
        try:
            embedding = await self._embedder.embed(query_text)
        except Exception as exc:  # noqa: BLE001 - degrade to no context
            logger.error("Embedding failed during search", extra={"error": str(exc)})
            return []
        if not embedding:
            return []
        return await self.search_by_embedding(embedding, top_k=top_k)

    async def search_by_embedding(
        self, embedding: Sequence[float], top_k: Optional[int] = None
    ) -> List[MemorySearchResult]:
        """Cosine-similarity search over an already-computed embedding."""
        limit = top_k or self.default_top_k
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, content, source, metadata, embedding <=> $1::vector AS distance
                    FROM knowledge_base
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    vector_to_literal(embedding),
                    limit,
                )
        except asyncpg.PostgresError as exc:
            raise MemoryStoreError(f"Memory search failed: {exc}") from exc

        results: List[MemorySearchResult] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                MemorySearchResult(
                    id=row["id"],
                    content=row["content"],
                    source=row["source"],
                    metadata=metadata or {},
                    distance=float(row["distance"]),
                )
            )
        return results

    # ── Maintenance ─────────────────────────────────────────────────────

    async def count(self) -> int:
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval("SELECT count(*) FROM knowledge_base")
        except asyncpg.PostgresError as exc:
            raise MemoryStoreError(f"Memory count failed: {exc}") from exc

    async def delete(self, memory_id: uuid.UUID) -> bool:
        try:
            async with self._pool.acquire() as conn:
                status = await conn.execute(
                    "DELETE FROM knowledge_base WHERE id = $1", memory_id
                )
            return status.endswith(" 1")
        except asyncpg.PostgresError as exc:
            raise MemoryStoreError(f"Memory delete failed: {exc}") from exc


async def retrieve_relevant_context(
    store: MemoryStore, query_text: str, top_k: Optional[int] = None
) -> List[str]:
    """Convenience helper: return just the ``content`` of the top-k matches."""
    results = await store.search(query_text, top_k=top_k)
    return [r.content for r in results]


__all__ = [
    "MemoryStore",
    "MemoryStoreError",
    "vector_to_literal",
    "retrieve_relevant_context",
]
