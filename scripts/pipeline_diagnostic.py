"""
Pipeline Diagnostic — tests every stage of the AI Calling Agent pipeline.

Stages tested:
  [0] Environment / Config
  [1] Twilio REST API  (credential + phone-number validation)
  [2] Groq LLM         (chat completion round-trip)
  [3] Deepgram TTS     (text → mulaw audio)
  [4] Deepgram STT     (WebSocket handshake)
  [5] Intent Classifier (heuristic node)
  [6] Memory Store     (pgvector — degrades gracefully)
  [7] LLM Greeting     (full greeting_node with LLM)
  [8] Full Agent Turn  (greeting → caller utterance → LLM response → intent)
  [9] REST API Health  (live server on localhost:8000)

Usage::

    python scripts/pipeline_diagnostic.py

Run while uvicorn is up for the REST API stage; the rest are standalone.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Bootstrap: add project root to sys.path ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PAD = 42  # label column width

results: List[Tuple[str, str, str]] = []  # (stage, status, detail)


def _hdr(title: str) -> None:
    print(f"\n{BOLD}{CYAN}" + "-"*60 + RESET)
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}" + "-"*60 + RESET)


def _ok(label: str, detail: str = "") -> None:
    tag = f"{GREEN}PASS{RESET}"
    print(f"  {label:<{PAD}} [{tag}]  {detail}")
    results.append((label, "PASS", detail))


def _fail(label: str, detail: str = "") -> None:
    tag = f"{RED}FAIL{RESET}"
    print(f"  {label:<{PAD}} [{tag}]  {detail}")
    results.append((label, "FAIL", detail))


def _warn(label: str, detail: str = "") -> None:
    tag = f"{YELLOW}WARN{RESET}"
    print(f"  {label:<{PAD}} [{tag}]  {detail}")
    results.append((label, "WARN", detail))


# Stage 0 - Config / Environment
def stage_config() -> Any:
    _hdr("Stage 0 - Config / Environment")
    from app.config import Settings

    try:
        cfg = Settings()
    except Exception as exc:
        _fail("Settings load", str(exc))
        raise SystemExit(1) from exc

    checks = {
        "GROQ_API_KEY":            cfg.groq_api_key,
        "DEEPGRAM_API_KEY":        cfg.deepgram_api_key,
        "TWILIO_ACCOUNT_SID":      cfg.twilio_account_sid,
        "TWILIO_AUTH_TOKEN":       cfg.twilio_auth_token,
        "TWILIO_PHONE_NUMBER":     cfg.twilio_phone_number,
        "TWILIO_WEBHOOK_BASE_URL": cfg.twilio_webhook_base_url,
        "AGENT_NAME":              cfg.agent_name,
        "COMPANY_NAME":            cfg.company_name,
        "GROQ_CHAT_MODEL":         cfg.groq_chat_model,
        "DEEPGRAM_TTS_MODEL":      cfg.deepgram_tts_model,
        "DEEPGRAM_STT_MODEL":      cfg.deepgram_stt_model,
    }

    for key, val in checks.items():
        if not val or val in ("", "gsk-unconfigured"):
            _fail(f"  {key}", "MISSING or placeholder")
        else:
            masked = val[:6] + "..." if len(val) > 8 else val
            _ok(f"  {key}", masked)

    _ok("Settings loaded", f"env={cfg.app_env}, port={cfg.app_port}")
    return cfg


# Stage 1 - Twilio REST
async def stage_twilio(cfg: Any) -> None:
    _hdr("Stage 1 - Twilio REST API")
    try:
        from twilio.rest import Client as TwilioSdkClient

        def _fetch():
            client = TwilioSdkClient(cfg.twilio_account_sid, cfg.twilio_auth_token)
            acct = client.api.accounts(cfg.twilio_account_sid).fetch()
            return acct.friendly_name, acct.status

        name, status = await asyncio.to_thread(_fetch)
        _ok("Account fetch", f"name={name!r}  status={status}")
    except Exception as exc:
        _fail("Account fetch", f"{type(exc).__name__}: {exc}")

    try:
        from twilio.rest import Client as TwilioSdkClient

        def _validate_number():
            client = TwilioSdkClient(cfg.twilio_account_sid, cfg.twilio_auth_token)
            numbers = client.incoming_phone_numbers.list(phone_number=cfg.twilio_phone_number)
            return len(numbers) > 0

        found = await asyncio.to_thread(_validate_number)
        if found:
            _ok("Twilio phone number", cfg.twilio_phone_number)
        else:
            _warn("Twilio phone number", f"{cfg.twilio_phone_number} not found under this account")
    except Exception as exc:
        _fail("Twilio phone number check", f"{type(exc).__name__}: {exc}")


# Stage 2 - Groq LLM
async def stage_llm(cfg: Any) -> None:
    _hdr("Stage 2 - Groq LLM (chat completions)")
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=cfg.groq_api_key, base_url=cfg.groq_base_url)
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=cfg.groq_chat_model,
            messages=[
                {"role": "system", "content": "You are a test assistant. Reply with exactly: PIPELINE_OK"},
                {"role": "user",   "content": "Reply with exactly: PIPELINE_OK"},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        reply = (resp.choices[0].message.content or "").strip()
        if "PIPELINE_OK" in reply:
            _ok("Groq LLM round-trip", f"model={cfg.groq_chat_model}  latency={latency:.0f}ms")
        else:
            _warn("Groq LLM round-trip", f"Unexpected reply: {reply!r}")
    except Exception as exc:
        _fail("Groq LLM round-trip", f"{type(exc).__name__}: {exc}")

    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=cfg.groq_api_key, base_url=cfg.groq_base_url)
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=cfg.groq_chat_model,
            messages=[
                {"role": "system", "content": "You are Angel, a calling agent for Jatin.Pvt.Ltd."},
                {"role": "user",   "content": "I want to reschedule my appointment."},
            ],
            max_tokens=100,
            temperature=0.4,
        )
        latency = (time.perf_counter() - t0) * 1000
        reply = (resp.choices[0].message.content or "").strip()
        _ok("Groq LLM realistic prompt", f"latency={latency:.0f}ms  reply_len={len(reply)}chars")
        print(f"    Reply preview: {reply[:120]!r}")
    except Exception as exc:
        _fail("Groq LLM realistic prompt", f"{type(exc).__name__}: {exc}")


# Stage 3 - Deepgram TTS
async def stage_tts(cfg: Any) -> None:
    _hdr("Stage 3 - Deepgram TTS (text to mu-law audio)")
    from app.voice.deepgram_client import DeepgramTTSClient, VoiceClientError

    tts = DeepgramTTSClient(cfg)
    test_text = "Hello Priya! This is Angel calling from Jatin Private Limited. How can I help you today?"
    try:
        t0 = time.perf_counter()
        audio = await tts.speak(test_text)
        latency = (time.perf_counter() - t0) * 1000
        if audio and len(audio) > 100:
            duration_ms = (len(audio) / 8000) * 1000
            _ok("TTS synthesis", f"bytes={len(audio)}  ~{duration_ms:.0f}ms audio  latency={latency:.0f}ms")
        else:
            _fail("TTS synthesis", f"Got {len(audio)} bytes -- too small or empty")
    except VoiceClientError as exc:
        _fail("TTS synthesis", f"VoiceClientError: {exc}")
    except Exception as exc:
        _fail("TTS synthesis", f"{type(exc).__name__}: {exc}")

    try:
        empty = await tts.speak("")
        if empty == b"":
            _ok("TTS empty string guard", "returns b'' as expected")
        else:
            _warn("TTS empty string guard", f"returned {len(empty)} bytes")
    except Exception as exc:
        _fail("TTS empty string guard", str(exc))


# Stage 4 - Deepgram STT WebSocket
async def stage_stt(cfg: Any) -> None:
    _hdr("Stage 4 - Deepgram STT (WebSocket handshake)")
    from app.voice.deepgram_client import DeepgramSTTClient, VoiceClientError

    stt = DeepgramSTTClient(cfg, handshake_timeout=5.0, flush_timeout=2.0)
    try:
        t0 = time.perf_counter()
        await stt.connect()
        latency = (time.perf_counter() - t0) * 1000
        _ok("STT WebSocket connect", f"connected in {latency:.0f}ms  model={cfg.deepgram_stt_model}")

        silence = bytes([0x7F] * 640)
        await stt.send_audio(silence)
        _ok("STT audio send", "20ms mu-law silence chunk sent")

        if stt._keepalive_task and not stt._keepalive_task.done():
            _ok("STT keepalive task", "running")
        else:
            _warn("STT keepalive task", "not running")

    except VoiceClientError as exc:
        _fail("STT WebSocket connect", f"VoiceClientError: {exc}")
    except Exception as exc:
        _fail("STT WebSocket connect", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        try:
            await stt.close()
            _ok("STT WebSocket close", "graceful close")
        except Exception as exc:
            _warn("STT WebSocket close", str(exc))


# Stage 5 - Intent Classifier
def stage_intent() -> None:
    _hdr("Stage 5 - Intent Classifier (heuristic)")
    from app.agent.nodes.query_handler import heuristic_classify

    test_cases = [
        ("I want to reschedule my appointment", "reschedule",         "neutral"),
        ("I need to cancel my booking",          "cancellation",       "neutral"),
        ("Goodbye, thanks!",                     "end_call",           "positive"),
        ("This is terrible service",             "general_inquiry",    "negative"),
        ("When is my appointment?",              "appointment_inquiry","neutral"),
        ("I need a manager please",              "general_inquiry",    "neutral"),
    ]

    for text, exp_intent, exp_sent in test_cases:
        got_intent, got_sent = heuristic_classify(text)
        intent_ok = got_intent == exp_intent
        sent_ok   = got_sent   == exp_sent
        if intent_ok and sent_ok:
            _ok(f"  '{text[:38]}'", f"intent={got_intent}  sentiment={got_sent}")
        else:
            detail = []
            if not intent_ok:
                detail.append(f"intent: expected={exp_intent} got={got_intent}")
            if not sent_ok:
                detail.append(f"sentiment: expected={exp_sent} got={got_sent}")
            _fail(f"  '{text[:38]}'", "  |  ".join(detail))

    from app.config import Settings
    cfg = Settings()
    keywords = cfg.escalation_keyword_list
    test_text = "I want to speak to a manager please"
    triggered = any(kw in test_text.lower() for kw in keywords)
    if triggered:
        _ok("  Escalation keyword trigger", f"keywords={keywords[:3]}")
    else:
        _warn("  Escalation keyword trigger", f"No match for: {test_text!r}")


# Stage 6 - Memory Store / pgvector
async def stage_memory(cfg: Any) -> None:
    _hdr("Stage 6 - Memory Store (pgvector)")
    try:
        import asyncpg
        conn = await asyncpg.connect(cfg.asyncpg_dsn, timeout=4.0)
        row = await conn.fetchrow("SELECT version()")
        await conn.close()
        _ok("PostgreSQL connection", str(row[0])[:60])

        conn2 = await asyncpg.connect(cfg.asyncpg_dsn, timeout=4.0)
        ext = await conn2.fetchrow(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        )
        await conn2.close()
        if ext:
            _ok("pgvector extension", f"version={ext[0]}")
        else:
            _warn("pgvector extension", "not installed -- memory features will be disabled")

    except Exception as exc:
        _warn("PostgreSQL connection", f"{type(exc).__name__}: {exc}")
        _warn("Memory store", "Degraded mode -- calls proceed without memory context")
        return

    try:
        from app.memory.embeddings import EmbeddingClient
        ec = EmbeddingClient(model_name=cfg.embedding_model, device=cfg.embedding_device)
        vec = ec.embed("appointment reminder for Priya Sharma")
        if len(vec) == 384:
            _ok("Embedding model", f"model={cfg.embedding_model}  dims={len(vec)}")
        else:
            _warn("Embedding model", f"unexpected dims={len(vec)}")
    except Exception as exc:
        _fail("Embedding model", f"{type(exc).__name__}: {exc}")


# Stage 7 - LLM Greeting Node
async def stage_greeting(cfg: Any) -> None:
    _hdr("Stage 7 - LLM Greeting Node")
    from app.api import dependencies
    from app.api.dependencies import GroqChatResponder

    responder = GroqChatResponder(cfg)
    dependencies.set_override("settings", cfg)
    dependencies.set_override("response_generator", responder)

    try:
        from app.agent.nodes.greeting import greeting_node

        state: Dict[str, Any] = {
            "call_sid": "diag-test-001",
            "conversation_history": [],
            "retrieved_memories": [],
            "caller_intent": None,
            "caller_sentiment": None,
            "should_escalate": False,
            "call_summary": None,
            "turn_count": 0,
            "metadata": {
                "context": {
                    "customer_name": "Priya Sharma",
                    "purpose": "appointment_reminder",
                    "appointment_time": "2025-10-01T10:00:00",
                }
            },
        }

        t0 = time.perf_counter()
        update = await greeting_node(state)
        latency = (time.perf_counter() - t0) * 1000

        history = update.get("conversation_history", [])
        if history:
            greeting_text = history[0].get("content", "")
            _ok("LLM greeting generated", f"latency={latency:.0f}ms  len={len(greeting_text)}chars")
            print(f"    Greeting: {greeting_text!r}")
            state["conversation_history"] = history
            update2 = await greeting_node(state)
            if update2 == {}:
                _ok("Greeting node idempotency", "returns {} when history exists")
            else:
                _warn("Greeting node idempotency", f"returned non-empty: {update2}")
        else:
            _fail("LLM greeting generated", "No conversation_history in update")
    except Exception as exc:
        _fail("LLM greeting generated", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        dependencies.clear_overrides()


# Stage 8 - Full Agent Turn
async def stage_full_agent_turn(cfg: Any) -> None:
    _hdr("Stage 8 - Full Agent Turn (Greeting -> Utterances -> LLM Reply)")
    from app.api import dependencies
    from app.api.dependencies import GroqChatResponder

    responder = GroqChatResponder(cfg)
    dependencies.set_override("settings", cfg)
    dependencies.set_override("response_generator", responder)

    try:
        from app.telephony.media_stream import CallOrchestrator

        orch = CallOrchestrator(
            call_sid="diag-test-002",
            settings=cfg,
            context={
                "customer_name": "Priya Sharma",
                "purpose": "appointment_reminder",
                "appointment_time": "2025-10-01T10:00:00",
            },
        )

        t0 = time.perf_counter()
        greetings = await orch.start_call()
        latency = (time.perf_counter() - t0) * 1000
        if greetings:
            _ok("Orchestrator.start_call()", f"latency={latency:.0f}ms")
            print(f"    Greeting: {greetings[0]!r}")
        else:
            _fail("Orchestrator.start_call()", "No greeting messages returned")

        test_utterances = [
            "Hi yes, I wanted to confirm my appointment",
            "Can you reschedule it to next Tuesday?",
        ]
        for utterance in test_utterances:
            t0 = time.perf_counter()
            outcome = await orch.on_caller_utterance(utterance)
            latency = (time.perf_counter() - t0) * 1000
            if outcome.agent_messages:
                _ok(
                    f"Utterance: '{utterance[:30]}...'",
                    f"latency={latency:.0f}ms  reply={len(outcome.agent_messages[0])}chars"
                )
                print(f"    Agent: {outcome.agent_messages[0]!r}")
            else:
                _warn(
                    f"Utterance: '{utterance[:30]}...'",
                    f"No agent messages (escalate={outcome.should_escalate} close={outcome.should_close})"
                )

        t0 = time.perf_counter()
        close_outcome = await orch.on_caller_utterance("Goodbye, thank you!")
        latency = (time.perf_counter() - t0) * 1000
        if close_outcome.should_close:
            _ok("End-call intent detection", f"should_close=True  latency={latency:.0f}ms")
        else:
            _warn("End-call intent detection", f"should_close={close_outcome.should_close}")

    except Exception as exc:
        _fail("Full agent turn", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        dependencies.clear_overrides()


# Stage 9 - REST API
async def stage_rest_api() -> None:
    _hdr("Stage 9 - REST API (localhost:8000)")
    base = "http://localhost:8000"

    try:
        req = urllib.request.urlopen(f"{base}/health", timeout=5)
        data = json.loads(req.read())
        _ok("GET /health", str(data))
    except Exception as exc:
        _fail("GET /health", f"{type(exc).__name__}: {exc}")
        _warn("REST API stages", "Server not reachable -- skipping remaining checks")
        return

    try:
        req = urllib.request.urlopen(f"{base}/openapi.json", timeout=5)
        spec = json.loads(req.read())
        routes = sorted(spec.get("paths", {}).keys())
        _ok("GET /openapi.json", f"{len(routes)} routes: {', '.join(routes)}")
    except Exception as exc:
        _fail("GET /openapi.json", str(exc))

    try:
        req = urllib.request.Request(f"{base}/calls", headers={"Accept": "application/json"})
        resp = urllib.request.urlopen(req, timeout=5)
        calls = json.loads(resp.read())
        _ok("GET /calls", f"{len(calls)} call(s) in-memory")
        for c in calls[:3]:
            ctx = c.get("context", {})
            print(f"    {c.get('call_sid','?')} | to={c.get('to_number')} | status={c.get('status')} | ctx={ctx}")
    except Exception as exc:
        _fail("GET /calls", str(exc))

    try:
        from app.config import Settings
        cfg = Settings()
        pub = cfg.twilio_webhook_base_url.rstrip("/") + "/health"
        req = urllib.request.Request(
            pub,
            headers={"ngrok-skip-browser-warning": "true", "Accept": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read())
        _ok("Public URL (ngrok) /health", f"{pub}  ->  {data}")
    except Exception as exc:
        _warn("Public URL (ngrok) /health", f"{type(exc).__name__}: {exc}")


# Summary
def print_summary() -> None:
    print(f"\n{BOLD}{CYAN}" + "="*60 + RESET)
    print(f"{BOLD}  PIPELINE DIAGNOSTIC SUMMARY{RESET}")
    print(f"{BOLD}{CYAN}" + "="*60 + RESET)

    passed = [r for r in results if r[1] == "PASS"]
    warned = [r for r in results if r[1] == "WARN"]
    failed = [r for r in results if r[1] == "FAIL"]

    print(f"  {GREEN}PASS:{RESET} {len(passed)}   {YELLOW}WARN:{RESET} {len(warned)}   {RED}FAIL:{RESET} {len(failed)}")

    if failed:
        print(f"\n{BOLD}{RED}  FAILURES:{RESET}")
        for label, _, detail in failed:
            print(f"    [FAIL]  {label}")
            if detail:
                print(f"            {detail}")

    if warned:
        print(f"\n{BOLD}{YELLOW}  WARNINGS:{RESET}")
        for label, _, detail in warned:
            print(f"    [WARN]  {label}")
            if detail:
                print(f"            {detail}")

    if not failed:
        print(f"\n  {GREEN}{BOLD}All critical checks passed!{RESET}")

    print(f"\n{BOLD}" + "="*60 + RESET + "\n")

    if failed:
        sys.exit(1)


async def main() -> None:
    print(f"{BOLD}{CYAN}")
    print("=" * 60)
    print("  AI CALLING AGENT -- FULL PIPELINE DIAGNOSTIC")
    print("=" * 60)
    print(RESET)

    cfg = stage_config()
    await stage_twilio(cfg)
    await stage_llm(cfg)
    await stage_tts(cfg)
    await stage_stt(cfg)
    stage_intent()
    await stage_memory(cfg)
    await stage_greeting(cfg)
    await stage_full_agent_turn(cfg)
    await stage_rest_api()
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
