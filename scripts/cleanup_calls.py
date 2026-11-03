"""Purge stale call records (e.g. stuck 'initiated' rows older than N days).

Usage::

    python scripts/cleanup_calls.py --days 7
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402
from sqlalchemy.sql import func  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.connection import create_engine, create_session_factory  # noqa: E402
from app.db.models import Call  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cleanup_calls")


async def cleanup(days: int, dry_run: bool) -> int:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            stale = select(Call.id).where(
                Call.status.in_(["initiated", "queued", "ringing"]),
                Call.started_at < func.now() - func.make_interval(0, 0, 0, days),
            )
            ids = (await session.execute(stale)).scalars().all()
            if dry_run:
                logger.info("Dry run: %d stale calls would be deleted", len(ids))
                return len(ids)
            if ids:
                await session.execute(delete(Call).where(Call.id.in_(ids)))
                await session.commit()
            logger.info("Deleted %d stale call records", len(ids))
            return len(ids)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="Age threshold in days")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(cleanup(args.days, args.dry_run))


if __name__ == "__main__":
    main()
