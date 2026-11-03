"""Async SQLAlchemy engine and session factory management.

SQLAlchemy (via the ``asyncpg`` dialect) is used for all relational work —
call records and transcripts. Raw vector queries bypass SQLAlchemy and use an
:mod:`asyncpg` connection pool directly (see :mod:`app.memory.store`), because
SQLAlchemy has no native pgvector expression support for ``<=>``.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from app.config import Settings

logger = logging.getLogger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    """Create an async SQLAlchemy engine from application settings."""
    return create_async_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=settings.log_level.upper() == "DEBUG",
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to ``engine``."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def check_database_health(engine: AsyncEngine) -> bool:
    """Run a trivial query to verify database connectivity."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - health check must never raise
        logger.warning("Database health check failed", extra={"error": str(exc)})
        return False


async def dispose_engine(engine: AsyncEngine) -> None:
    """Dispose of the engine's connection pool gracefully."""
    try:
        await engine.dispose()
        logger.info("Database engine disposed")
    except Exception as exc:  # noqa: BLE001 - shutdown must not crash the app
        logger.warning("Error while disposing database engine", extra={"error": str(exc)})


__all__ = [
    "create_engine",
    "create_session_factory",
    "check_database_health",
    "dispose_engine",
]
