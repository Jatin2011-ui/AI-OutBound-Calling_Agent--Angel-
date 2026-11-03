"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.api import dependencies
from app.db.connection import check_database_health

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness probe; includes DB reachability when configured."""
    db_ok: bool | None = None
    try:
        engine = dependencies.get_engine()
        db_ok = await check_database_health(engine)
    except RuntimeError:
        db_ok = None  # DB not configured in this deployment
    return {"status": "ok", "db": db_ok}


__all__ = ["router"]
