"""Seed the pgvector knowledge base from a JSONL file.

Usage::

    python scripts/seed_memory.py --input data/knowledge_base.jsonl

Each line must look like::

    {"content": "...", "source": "policy_doc", "metadata": {"category": "policy"}}

Documents are embedded in batches of up to 100 and inserted into
``knowledge_base``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import asyncpg

# Allow running as ``python scripts/seed_memory.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.memory.embeddings import EMBEDDING_BATCH_SIZE, EmbeddingClient  # noqa: E402
from app.memory.schemas import MemoryRecordCreate  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_memory")


def load_records(path: Path) -> list[MemoryRecordCreate]:
    records: list[MemoryRecordCreate] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(MemoryRecordCreate.model_validate_json(line))
            except ValueError as exc:
                logger.warning("Skipping invalid JSONL line %d: %s", line_number, exc)
    return records


async def seed(input_path: Path, batch_size: int = EMBEDDING_BATCH_SIZE) -> int:
    settings = get_settings()
    records = load_records(input_path)
    if not records:
        logger.warning("No records found in %s", input_path)
        return 0

    pool = await asyncpg.create_pool(dsn=settings.asyncpg_dsn, min_size=1, max_size=4)
    try:
        embedder = EmbeddingClient(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        store = MemoryStore(pool=pool, embedder=embedder)

        inserted = 0
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            # Embed the whole batch in one request, then insert one by one.
            embeddings = await embedder.embed_many([r.content for r in batch])
            for record, embedding in zip(batch, embeddings):
                if not embedding:
                    logger.warning("Empty embedding; skipping: %s", record.content[:60])
                    continue
                from app.memory.store import vector_to_literal

                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO knowledge_base (content, embedding, source, metadata)
                        VALUES ($1, $2::vector, $3, $4::jsonb)
                        """,
                        record.content,
                        vector_to_literal(embedding),
                        record.source,
                        json.dumps(record.metadata),
                    )
                inserted += 1
            logger.info("Seeded batch %d–%d", start, start + len(batch) - 1)
        await embedder.close()
        return inserted
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to JSONL seed file")
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    inserted = asyncio.run(seed(args.input, batch_size=args.batch_size))
    logger.info("Done — %d records seeded", inserted)


if __name__ == "__main__":
    main()
