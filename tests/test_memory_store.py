"""Tests for the embedding client and pgvector memory store.

No real database or model downloads: the asyncpg pool is faked and the
sentence-transformers model is replaced with an in-memory fake.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, List

from app.memory.embeddings import EmbeddingClient
from app.memory.schemas import MemoryRecordCreate
from app.memory.store import MemoryStore, vector_to_literal

# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeAsyncpgConnection:
    def __init__(self) -> None:
        self.queries: List[tuple] = []
        self.fetch_result: List[dict] = []
        self.fetchrow_result: Any = None
        self.fetchval_result: Any = 0

    async def fetch(self, query: str, *args: Any) -> List[dict]:
        self.queries.append((query, args))
        return self.fetch_result

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.queries.append((query, args))
        return self.fetchrow_result

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.queries.append((query, args))
        return self.fetchval_result


class _AcquireContext:
    def __init__(self, conn: FakeAsyncpgConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeAsyncpgConnection:
        return self._conn

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


class FakeAsyncpgPool:
    def __init__(self, conn: FakeAsyncpgConnection) -> None:
        self.conn = conn

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.conn)


class FakeEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [0.1] * 1536
        self.calls: List[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.vector)


# ── Upsert ───────────────────────────────────────────────────────────────────


async def test_upsert_inserts_vector_cast_and_returns_id():
    new_id = uuid.uuid4()
    conn = FakeAsyncpgConnection()
    conn.fetchrow_result = {"id": new_id}
    store = MemoryStore(pool=FakeAsyncpgPool(conn), embedder=FakeEmbedder())  # type: ignore[arg-type]

    record = MemoryRecordCreate(
        content="Cancellation is free 24h ahead.", source="policy", metadata={"category": "policy"}
    )
    result = await store.upsert(record)

    assert result == new_id
    query, args = conn.queries[0]
    assert "INSERT INTO knowledge_base" in query
    assert "$2::vector" in query
    assert args[0] == record.content
    assert args[1].startswith("[0.10000000")
    assert json.loads(args[3]) == {"category": "policy"}


async def test_upsert_skips_when_embedding_fails():
    conn = FakeAsyncpgConnection()
    store = MemoryStore(pool=FakeAsyncpgPool(conn), embedder=FakeEmbedder(vector=[]))  # type: ignore[arg-type]
    result = await store.upsert(MemoryRecordCreate(content="will not embed"))
    assert result is None
    assert conn.queries == []


# ── Search ───────────────────────────────────────────────────────────────────


async def test_search_runs_cosine_query_and_maps_rows():
    conn = FakeAsyncpgConnection()
    row_id = uuid.uuid4()
    conn.fetch_result = [
        {
            "id": row_id,
            "content": "Reschedule up to 24 hours in advance.",
            "source": "policy",
            "metadata": {"category": "policy"},
            "distance": 0.05,
        }
    ]
    store = MemoryStore(
        pool=FakeAsyncpgPool(conn), embedder=FakeEmbedder(), default_top_k=5  # type: ignore[arg-type]
    )

    results = await store.search("until when can I reschedule?")

    query, args = conn.queries[0]
    assert "embedding <=> $1::vector" in query
    assert args[1] == 5  # default top_k honoured
    assert len(results) == 1
    assert results[0].id == row_id
    assert results[0].distance == 0.05
    assert results[0].metadata == {"category": "policy"}


async def test_search_without_embedder_returns_empty():
    conn = FakeAsyncpgConnection()
    store = MemoryStore(pool=FakeAsyncpgPool(conn), embedder=None)  # type: ignore[arg-type]
    assert await store.search("anything") == []
    assert conn.queries == []


# ── Embedding client batching (local sentence-transformers) ─────────────────


class FakeSentenceTransformer:
    """Mimics ``SentenceTransformer.encode`` without PyTorch/model downloads."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.batch_sizes: List[int] = []

    def encode(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        self.batch_sizes.append(len(texts))
        return [[0.5] * self.dimension for _ in texts]


async def test_embed_many_chunks_batches_of_100():
    fake_model = FakeSentenceTransformer(dimension=4)
    client = EmbeddingClient(model=fake_model)

    vectors = await client.embed_many([f"doc {i}" for i in range(250)])

    assert fake_model.batch_sizes == [100, 100, 50]
    assert len(vectors) == 250
    assert all(len(v) == 4 for v in vectors)


async def test_embed_failure_degrades_to_empty_vector():
    class ExplodingModel:
        def encode(self, texts: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("model unavailable")

    client = EmbeddingClient(model=ExplodingModel())
    assert await client.embed("anything") == []


def test_vector_to_literal_formatting():
    assert vector_to_literal([0.1, -0.25, 3]) == "[0.10000000,-0.25000000,3.00000000]"
