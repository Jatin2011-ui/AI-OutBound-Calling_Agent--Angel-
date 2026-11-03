"""Async embedding generation with local HuggingFace sentence-transformers.

Default model: ``all-MiniLM-L6-v2`` (384-dimensional vectors). The
``sentence-transformers`` encode call is synchronous and CPU-bound, so it is
offloaded to a worker thread via :func:`asyncio.to_thread` to keep the event
loop responsive during live calls. The model is loaded lazily on first use.

The ``model`` constructor parameter accepts any pre-built object with an
``encode(texts)`` method, which is how tests inject a fake without touching
PyTorch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 100
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingError(Exception):
    """Raised when embedding generation ultimately fails."""


class EmbeddingClient:
    """Async wrapper around a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        model: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._model = model  # pre-built / injected model (used by tests)

    # ── Model lifecycle ─────────────────────────────────────────────────

    def _ensure_model(self) -> Any:
        """Load the sentence-transformers model on first use (blocking)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model", extra={"model": self.model_name}
            )
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    async def close(self) -> None:
        """Release the model reference (kept symmetrical with other clients)."""
        self._model = None

    # ── Encoding ────────────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns an empty list on failure."""
        results = await self.embed_many([text])
        return results[0] if results else []

    async def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed many strings, chunking work into batches of 100.

        Input order is preserved. A batch that fails yields empty vectors for
        its inputs so callers can degrade gracefully instead of crashing a
        live call.
        """
        cleaned = [t if t and t.strip() else " " for t in texts]
        vectors: List[List[float]] = []
        for start in range(0, len(cleaned), EMBEDDING_BATCH_SIZE):
            batch = cleaned[start : start + EMBEDDING_BATCH_SIZE]
            if not batch:
                continue
            vectors.extend(await self._encode_batch(batch))
        return vectors

    async def _encode_batch(self, batch: List[str]) -> List[List[float]]:
        try:
            model = await asyncio.to_thread(self._ensure_model)
            encoded = await asyncio.to_thread(model.encode, batch)
            return [[float(v) for v in vector] for vector in encoded]
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.error(
                "Embedding batch failed",
                extra={"error": str(exc), "batch_size": len(batch)},
            )
            return [[] for _ in batch]


__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "DEFAULT_MODEL_NAME",
]
