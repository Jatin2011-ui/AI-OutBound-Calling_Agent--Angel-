"""Call management endpoints: initiate outbound calls and fetch call records."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api import dependencies
from app.telephony.twilio_client import TelephonyError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["calls"])


class InitiateCallRequest(BaseModel):
    """Payload for ``POST /calls/initiate``."""

    to_number: str = Field(
        pattern=r"^\+[1-9]\d{6,14}$",
        description="Destination number in E.164 format.",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Call context (customer_name, purpose, ...) for the agent.",
    )


class InitiateCallResponse(BaseModel):
    call_id: str
    call_sid: str
    status: str
    to_number: str


@router.post(
    "/initiate",
    response_model=InitiateCallResponse,
    status_code=status.HTTP_201_CREATED,
)
async def initiate_call(body: InitiateCallRequest) -> InitiateCallResponse:
    """Place an outbound call via Twilio and record it."""
    settings = dependencies.get_settings()
    twilio = dependencies.get_twilio_client()

    try:
        handle = await twilio.initiate_call(
            to_number=body.to_number,
            twiml_url=settings.twiml_callback_url(body.context),
            status_callback_url=settings.status_callback_url(),
        )
    except TelephonyError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    repository = dependencies.get_call_repository()
    try:
        call_id = await repository.record_initiated(
            call_sid=handle.call_sid,
            to_number=handle.to_number,
            from_number=handle.from_number,
            context=body.context,
        )
    except Exception as exc:  # noqa: BLE001 - the call itself already succeeded
        logger.error(
            "Failed to persist call record",
            extra={"call_sid": handle.call_sid, "error": str(exc)},
        )
        call_id = str(uuid.uuid4())

    return InitiateCallResponse(
        call_id=call_id,
        call_sid=handle.call_sid,
        status=handle.status,
        to_number=handle.to_number,
    )


@router.get("", summary="List all calls with their context")
async def list_calls() -> list:
    """Return all recorded calls with their context, status, and summary."""
    repository = dependencies.get_call_repository()
    return await repository.list_calls()


@router.get("/{call_id}")
async def get_call(call_id: str) -> Dict[str, Any]:
    """Return the stored call record, transcript, and summary."""
    try:
        uuid.UUID(call_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="call_id must be a valid UUID",
        ) from exc

    repository = dependencies.get_call_repository()
    record = await repository.get_call(call_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Call not found"
        )
    return record


__all__ = ["router", "InitiateCallRequest", "InitiateCallResponse"]
