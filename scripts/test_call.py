"""Trigger a test outbound call against a running server.

Usage::

    python scripts/test_call.py --to +15551234567 --scenario appointment_reminder

Requires the FastAPI server to be running locally (default port 8000).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_CONTEXTS = {
    "appointment_reminder": {
        "customer_name": "Priya Sharma",
        "purpose": "appointment_reminder",
        "appointment_time": "2025-10-01T10:00:00",
    },
    "feedback_survey": {
        "customer_name": "Priya Sharma",
        "purpose": "feedback_survey",
    },
}


async def trigger_call(base_url: str, to_number: str, context: dict) -> None:
    payload = {"to_number": to_number, "context": context}
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        response = await client.post("/calls/initiate", json=payload)
    print(f"HTTP {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="Destination number (E.164)")
    parser.add_argument(
        "--scenario",
        default="appointment_reminder",
        choices=sorted(DEFAULT_CONTEXTS),
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    asyncio.run(trigger_call(args.base_url, args.to, DEFAULT_CONTEXTS[args.scenario]))


if __name__ == "__main__":
    main()
