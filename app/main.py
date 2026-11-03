"""FastAPI application entrypoint.

Run with::

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api import dependencies
from app.api.routes.calls import router as calls_router
from app.api.routes.health import router as health_router
from app.api.routes.webhooks import http_router as twilio_http_router
from app.api.routes.webhooks import ws_router as twilio_ws_router

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Root logging configuration with single-line structured-ish output."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: configure logging + initialise services. Shutdown: close them."""
    settings = dependencies.get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "Starting AI outbound calling agent",
        extra={"env": settings.app_env, "port": settings.app_port},
    )
    if settings.is_test:
        logger.info("Test environment detected — skipping service initialisation")
    else:
        await dependencies.initialise(settings)
    yield
    await dependencies.shutdown()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="AI Outbound Calling Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(calls_router, prefix="/calls")
    app.include_router(twilio_http_router)
    app.include_router(twilio_ws_router)
    return app


app = create_app()


__all__ = ["app", "create_app", "configure_logging"]
