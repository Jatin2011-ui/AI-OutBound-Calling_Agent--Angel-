"""Twilio REST API client for outbound call lifecycle management.

The official ``twilio`` SDK is synchronous, so all blocking calls are run in
a thread via :func:`asyncio.to_thread` to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client as TwilioSdkClient

logger = logging.getLogger(__name__)


class TelephonyError(Exception):
    """Raised when a Twilio API operation fails."""


@dataclass(frozen=True)
class CallHandle:
    """Minimal handle for an in-flight Twilio call."""

    call_sid: str
    status: str
    to_number: str
    from_number: str


class TwilioCallClient:
    """Async facade over the Twilio Voice API."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str) -> None:
        self._from_number = from_number
        # The SDK validates credentials lazily at request time.
        self._client = TwilioSdkClient(account_sid, auth_token)

    async def initiate_call(
        self,
        to_number: str,
        twiml_url: Optional[str] = None,
        status_callback_url: Optional[str] = None,
    ) -> CallHandle:
        """Place an outbound call whose TwiML is served from ``twiml_url``."""
        url = twiml_url or "about:blank"

        def _create() -> CallHandle:
            call = self._client.calls.create(
                to=to_number,
                from_=self._from_number,
                url=url,
                status_callback=status_callback_url,
                status_callback_event=["initiated", "ringing", "answered", "completed"],
                status_callback_method="POST",
            )
            return CallHandle(
                call_sid=call.sid,
                status=call.status,
                to_number=to_number,
                from_number=self._from_number,
            )

        try:
            handle = await asyncio.to_thread(_create)
            logger.info(
                "Outbound call initiated",
                extra={"call_sid": handle.call_sid, "to_number": to_number},
            )
            return handle
        except TwilioRestException as exc:
            logger.error(
                "Failed to initiate call",
                extra={"to_number": to_number, "error": str(exc)},
            )
            raise TelephonyError(f"Could not initiate call to {to_number}: {exc}") from exc

    async def end_call(self, call_sid: str) -> bool:
        """Hang up an in-progress call. Returns True on success."""

        def _update() -> bool:
            call = self._client.calls(call_sid).update(status="completed")
            return call.status in {"completed", "canceled", "busy", "failed", "no-answer"}

        try:
            result = await asyncio.to_thread(_update)
            logger.info("Call hangup requested", extra={"call_sid": call_sid})
            return result
        except TwilioRestException as exc:
            logger.warning(
                "Failed to end call (it may already be over)",
                extra={"call_sid": call_sid, "error": str(exc)},
            )
            return False

    async def fetch_call_status(self, call_sid: str) -> Optional[str]:
        """Fetch the current Twilio status for a call."""

        def _fetch() -> str:
            return self._client.calls(call_sid).fetch().status

        try:
            return await asyncio.to_thread(_fetch)
        except TwilioRestException as exc:
            logger.warning(
                "Failed to fetch call status",
                extra={"call_sid": call_sid, "error": str(exc)},
            )
            return None


__all__ = ["TwilioCallClient", "CallHandle", "TelephonyError"]
