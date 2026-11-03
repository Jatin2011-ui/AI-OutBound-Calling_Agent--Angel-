# 📞 AI Outbound Calling Agent

> A fully autonomous, voice-enabled outbound calling agent that orchestrates real-time telephone conversations using Groq (`openai/gpt-oss-120b`) for reasoning, Deepgram for streaming speech-to-text and text-to-speech, Twilio's programmable voice platform, local HuggingFace embeddings, and a pgvector-backed long-term memory store.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange.svg)](https://console.groq.com/docs)
[![Twilio](https://img.shields.io/badge/Twilio-Voice_API-red.svg)](https://www.twilio.com/docs/voice)
[![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-blue.svg)](https://github.com/pgvector/pgvector)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [System Design](#system-design)
  - [State Machine](#state-machine)
  - [LangGraph Agent Flow](#langgraph-agent-flow)
  - [Memory Architecture](#memory-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Agent Node Reference](#agent-node-reference)
- [Call Flow Walkthrough](#call-flow-walkthrough)
- [Memory & RAG Pipeline](#memory--rag-pipeline)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project demonstrates a production-grade **agentic AI system** capable of autonomously conducting outbound phone calls. The agent can:

- **Initiate outbound calls** via Twilio's programmable voice API
- **Listen and respond in real-time** using Deepgram streaming STT + TTS and Groq LLM replies
- **Remember context** across multi-turn dialogue via a pgvector semantic memory store
- **Navigate complex call flows** using a LangGraph-powered state machine: greeting → query handling → escalation → closing
- **Adapt dynamically** to conversation direction without hardcoded scripts

This system bridges the gap between LLM reasoning capabilities and real-world telephony infrastructure — a highly demanded skill in the 2025 AI engineering landscape.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AI Outbound Calling Agent                     │
│                                                                      │
│  ┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐  │
│  │   FastAPI   │────▶│  LangGraph Agent │────▶│  Deepgram + Groq  │  │
│  │   Server    │     │  (Orchestrator)  │     │      API         │  │
│  └──────┬──────┘     └────────┬─────────┘     └──────────────────┘  │
│         │                     │                                      │
│         │                     │ State Transitions                    │
│         │                     ▼                                      │
│  ┌──────▼──────┐     ┌──────────────────┐     ┌──────────────────┐  │
│  │   Twilio    │────▶│  State Machine   │────▶│   pgvector DB    │  │
│  │ Voice/Media │     │ (Call Flow Mgr)  │     │ (Memory + RAG)   │  │
│  │   Stream    │     └──────────────────┘     └──────────────────┘  │
│  └─────────────┘                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**

1. Trigger → FastAPI receives call initiation request
2. Twilio places outbound call and opens a Media Stream WebSocket
3. Audio flows Twilio → Deepgram STT → Groq LLM (via LangGraph) → Deepgram TTS → Twilio
4. LangGraph orchestrates agent nodes based on the current call state
5. Each turn, the memory node performs pgvector similarity search for relevant context
6. Agent generates a response, which is streamed back as audio via Twilio

---

## Features

| Feature | Description |
|---|---|
| 🎙️ **Real-time voice** | Streaming recognition + synthesis via Deepgram (nova-2 STT, aura-2 TTS) |
| 📞 **Outbound calling** | Programmatic call initiation and bidirectional audio via Twilio |
| 🧠 **Semantic memory** | pgvector-powered retrieval of relevant past interactions and knowledge |
| 🔄 **State machine** | Robust call flow management across greeting, query, escalation, and close states |
| 🕸️ **LangGraph orchestration** | Graph-based agent workflow with conditional routing and cycle detection |
| 📝 **Conversation history** | Per-call turn tracking persisted to PostgreSQL |
| 🔁 **Dynamic adaptation** | Agent adjusts strategy based on caller responses and context |
| 🚨 **Escalation logic** | Automatic handoff to human agents on defined triggers |
| 📊 **Call metadata** | Duration, outcome, sentiment, and transcript storage |
| 🔒 **Async-first** | Fully async Python architecture for high concurrency |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Web Framework** | FastAPI + Uvicorn |
| **Agent Orchestration** | LangGraph 0.2+ |
| **LLM (reasoning)** | Groq — `openai/gpt-oss-120b` (chat completions) |
| **Speech-to-Text** | Deepgram streaming STT (`nova-2`, 8 kHz μ-law WebSocket) |
| **Text-to-Speech** | Deepgram Aura TTS (`aura-2-asteria-en`, raw μ-law output) |
| **Voice / Telephony** | Twilio Programmable Voice + Media Streams |
| **Vector Database** | PostgreSQL + pgvector extension |
| **ORM** | SQLAlchemy (async) + asyncpg |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, sentence-transformers) |
| **Audio Processing** | PyAudio / audioop (μ-law ↔ PCM conversion) |
| **Task Queue** | asyncio (native) |
| **Config Management** | Pydantic Settings + python-dotenv |
| **Testing** | pytest + pytest-asyncio |
| **Containerization** | Docker + Docker Compose |

---

## Project Structure

```
ai-outbound-calling-agent/
│
├── README.md
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
│
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entrypoint
│   ├── config.py                   # Pydantic settings
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── calls.py            # POST /calls/initiate, GET /calls/{id}
│   │   │   ├── webhooks.py         # Twilio TwiML + Media Stream endpoints
│   │   │   └── health.py           # Health check endpoint
│   │   └── dependencies.py         # FastAPI dependency injection
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py                # LangGraph StateGraph definition
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── greeting.py         # Node: initial call greeting
│   │   │   ├── query_handler.py    # Node: main query resolution
│   │   │   ├── memory_retrieval.py # Node: pgvector similarity search
│   │   │   ├── response_gen.py     # Node: LLM response generation
│   │   │   ├── escalation.py       # Node: human handoff logic
│   │   │   └── closing.py          # Node: call wrap-up and summary
│   │   └── edges/
│   │       ├── __init__.py
│   │       └── conditions.py       # Conditional edge routing logic
│   │
│   ├── telephony/
│   │   ├── __init__.py
│   │   ├── twilio_client.py        # Outbound call initiation
│   │   ├── media_stream.py         # WebSocket audio stream handler
│   │   ├── twiml_builder.py        # TwiML response generation
│   │   └── audio_utils.py          # μ-law ↔ PCM conversion utilities
│   │
│   ├── voice/
│   │   ├── __init__.py
│   │   └── deepgram_client.py      # Deepgram streaming STT (WS) + TTS (REST)
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py                # pgvector read/write operations
│   │   ├── embeddings.py           # Local MiniLM embedding generation
│   │   └── schemas.py              # Memory record Pydantic models
│   │
│   └── db/
│       ├── __init__.py
│       ├── connection.py           # SQLAlchemy async engine + session
│       ├── models.py               # ORM models (Call, Memory, Transcript)
│       └── migrations/
│           └── 001_initial.sql     # pgvector schema setup
│
├── scripts/
│   ├── seed_memory.py              # Seed pgvector with knowledge base docs
│   ├── test_call.py                # Trigger a test outbound call
│   └── cleanup_calls.py            # Purge stale call records
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_agent_graph.py
    ├── test_memory_store.py
    ├── test_twilio_webhook.py
    ├── test_deepgram_client.py
    └── test_media_stream.py
```

---

## System Design

### State Machine

The call flow is modeled as a deterministic state machine with five states:

```
                    ┌───────────┐
         START ────▶│  GREETING │
                    └─────┬─────┘
                          │ caller responds
                          ▼
                    ┌───────────┐
              ┌────▶│  QUERYING │◀────┐
              │     └─────┬─────┘     │
              │           │           │ follow-up
              │    resolve│           │ question
              │           ▼           │
              │     ┌──────────┐      │
              │     │ RESPOND  │──────┘
              │     └─────┬────┘
              │           │ complex / unresolvable
              │           ▼
              │     ┌────────────┐
              │     │ ESCALATION │
              │     └─────┬──────┘
              │           │
              └───── human takes over ────▶ ┌─────────┐
                                            │ CLOSING │──▶ END
                                            └─────────┘
```

| State | Description | Transitions To |
|---|---|---|
| `GREETING` | Agent introduces itself and purpose | `QUERYING` |
| `QUERYING` | Extracts caller intent; retrieves memory context | `RESPONDING` |
| `RESPONDING` | Generates and delivers LLM response via audio | `QUERYING`, `CLOSING`, `ESCALATION` |
| `ESCALATION` | Flags call for human review; plays hold message | `CLOSING` |
| `CLOSING` | Thanks caller, ends call, persists summary | `END` |
| `INTERRUPTED` | Barge-in detected; agent stopped speaking, returned to listening | `LISTENING` (if caller continues) or `CLOSING` (if goodbye) |

---

### LangGraph Agent Flow

```python
# app/agent/graph.py — simplified overview

from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    greeting_node,
    memory_retrieval_node,
    query_handler_node,
    response_gen_node,
    escalation_node,
    closing_node,
)
from app.agent.edges.conditions import route_after_response

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("greeting",          greeting_node)
graph.add_node("memory_retrieval",  memory_retrieval_node)
graph.add_node("query_handler",     query_handler_node)
graph.add_node("response_gen",      response_gen_node)
graph.add_node("escalation",        escalation_node)
graph.add_node("closing",           closing_node)

# Define edges
graph.set_entry_point("greeting")
graph.add_edge("greeting",         "memory_retrieval")
graph.add_edge("memory_retrieval", "query_handler")
graph.add_edge("query_handler",    "response_gen")
graph.add_conditional_edges(
    "response_gen",
    route_after_response,
    {
        "continue":  "memory_retrieval",
        "escalate":  "escalation",
        "close":     "closing",
    }
)
graph.add_edge("escalation", "closing")
graph.add_edge("closing",    END)

agent = graph.compile()
```

**`AgentState` TypedDict** (tracks the full call context across nodes):

```python
# app/agent/state.py

from typing import TypedDict, List, Optional, Literal
from datetime import datetime

class Message(TypedDict):
    role: Literal["agent", "caller"]
    content: str
    timestamp: datetime

class AgentState(TypedDict):
    call_sid: str                        # Twilio Call SID
    call_state: str                      # Current FSM state
    conversation_history: List[Message]  # Full turn history
    retrieved_memories: List[str]        # pgvector search results
    caller_intent: Optional[str]         # Extracted intent
    caller_sentiment: Optional[str]      # positive / neutral / negative
    should_escalate: bool                # Escalation flag
    call_summary: Optional[str]          # Generated at closing
    turn_count: int                      # Number of completed turns
    metadata: dict                       # Call metadata (phone, time, etc.)
```

---

### Memory Architecture

The agent uses a two-layer memory model:

```
┌────────────────────────────────────────────────┐
│                  MEMORY LAYERS                 │
│                                                │
│  Layer 1: Short-term (In-process)              │
│  └─ AgentState.conversation_history            │
│     Per-call turn buffer (lost on call end)    │
│                                                │
│  Layer 2: Long-term (pgvector)                 │
│  └─ knowledge_base table                       │
│     ├─ Product/service knowledge docs          │
│     ├─ FAQ embeddings                          │
│     └─ Past resolved interaction summaries     │
└────────────────────────────────────────────────┘
```

At the `memory_retrieval` node, the agent embeds the caller's latest utterance and runs a cosine similarity search:

```sql
-- Top-5 relevant memory chunks
SELECT content, metadata
FROM knowledge_base
ORDER BY embedding <=> $1   -- pgvector cosine operator
LIMIT 5;
```

Retrieved chunks are injected into the system prompt for the response generation step.

---

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with `pgvector` extension
- Docker & Docker Compose (recommended)
- [Groq API key](https://console.groq.com/keys) (LLM reasoning — `openai/gpt-oss-120b`)
- [Deepgram API key](https://console.deepgram.com/) (STT + TTS)
- [Twilio account](https://www.twilio.com/try-twilio) with a phone number
- No other AI keys needed — embeddings run locally (HuggingFace MiniLM)
- `ngrok` or equivalent tunnel (for local Twilio webhook development)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/ai-outbound-calling-agent.git
cd ai-outbound-calling-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Copy environment variables

```bash
cp .env.example .env
# Edit .env with your credentials (see Configuration section)
```

---

## Configuration

All configuration is managed via environment variables. Copy `.env.example` to `.env` and populate:

```env
# ─── Application ───────────────────────────────────────────────────────────
APP_ENV=development                    # development | production
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO

# ─── Groq (LLM: chat completions / reasoning) ──────────────────────────────
GROQ_API_KEY=gsk-...
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_CHAT_MODEL=openai/gpt-oss-120b

# ─── Embeddings (local HuggingFace sentence-transformers — no API key) ─────
EMBEDDING_MODEL=all-MiniLM-L6-v2       # produces 384-dim vectors
EMBEDDING_DEVICE=cpu                   # cpu | cuda

# ─── Deepgram (speech: streaming STT + TTS) ────────────────────────────────
DEEPGRAM_API_KEY=...
DEEPGRAM_STT_WS_URL=wss://api.deepgram.com/v1/listen
DEEPGRAM_STT_MODEL=nova-2
DEEPGRAM_TTS_URL=https://api.deepgram.com/v1/speak
DEEPGRAM_TTS_MODEL=aura-2-asteria-en

# ─── Twilio ────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1XXXXXXXXXX       # Your Twilio number (E.164 format)
TWILIO_WEBHOOK_BASE_URL=https://your-ngrok-url.ngrok.io   # Public URL for webhooks

# ─── PostgreSQL + pgvector ─────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/calling_agent
PGVECTOR_DIMENSION=384                 # all-MiniLM-L6-v2 output dims

# ─── Agent Behavior ────────────────────────────────────────────────────────
AGENT_NAME=Alex                        # Agent's spoken name
AGENT_SYSTEM_PROMPT_FILE=prompts/system.txt
MEMORY_RETRIEVAL_TOP_K=5
MAX_TURNS_BEFORE_CLOSE=20
ESCALATION_KEYWORDS=manager,supervisor,complaint,cancel

# ─── Barge-in / Interruption ───────────────────────────────────────────────
BARGE_IN_ENABLED=True                  # enable/disable barge-in detection
BARGE_IN_MIN_SPEECH_MS=300             # ignore sounds shorter than this (ms)
BARGE_IN_TRIGGER=speech_started        # speech_started | interim | both
BARGE_IN_FILLER_TOKENS=uh-huh,uh huh,mm-hmm,mmhmm,yeah,yep,mm,uh,ah,hm,hmm
BARGE_IN_VAD_EVENTS=True               # request Deepgram SpeechStarted events

# ─── Escalation Webhook ────────────────────────────────────────────────────
ESCALATION_WEBHOOK_URL=                # Optional: Slack/Teams webhook for human handoff

---

## Database Setup

### Option A: Docker Compose (Recommended)

```bash
docker-compose up -d postgres
```

This spins up PostgreSQL 15 with `pgvector` pre-installed.

### Option B: Manual Setup

```bash
# Install pgvector extension (requires PostgreSQL 15+)
psql -U postgres -c "CREATE DATABASE calling_agent;"
psql -U postgres -d calling_agent -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Run migrations

```bash
psql -U postgres -d calling_agent -f app/db/migrations/001_initial.sql
```

**Schema overview:**

```sql
-- app/db/migrations/001_initial.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Stores outbound call records
CREATE TABLE calls (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_sid    TEXT UNIQUE NOT NULL,          -- Twilio Call SID
    to_number   TEXT NOT NULL,
    from_number TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'initiated',
    started_at  TIMESTAMPTZ DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    duration_s  INTEGER,
    summary     TEXT,
    metadata    JSONB DEFAULT '{}'
);

-- Per-turn conversation transcript
CREATE TABLE transcripts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id     UUID REFERENCES calls(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,                 -- 'agent' | 'caller'
    content     TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Semantic memory / knowledge base
CREATE TABLE knowledge_base (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    embedding   vector(384),                   -- pgvector column (MiniLM)
    source      TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX ON knowledge_base
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### Seed the knowledge base

```bash
python scripts/seed_memory.py --input data/knowledge_base.jsonl
```

The seeder script reads JSONL documents, generates embeddings, and upserts them into `knowledge_base`.

---

## Running the Application

### Development (local)

```bash
# Terminal 1: Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Expose local server to Twilio via ngrok
ngrok http 8000
# Copy the https://xxxxx.ngrok.io URL → set TWILIO_WEBHOOK_BASE_URL in .env
```

### Docker Compose (full stack)

```bash
docker-compose up --build
```

Starts: FastAPI app + PostgreSQL with pgvector.

### Trigger a test call

```bash
python scripts/test_call.py --to +91XXXXXXXXXX --scenario "appointment_reminder"
```

---

## API Reference

### `POST /calls/initiate`

Initiates an outbound call.

**Request body:**
```json
{
  "to_number": "+91XXXXXXXXXX",
  "context": {
    "customer_name": "Priya Sharma",
    "purpose": "appointment_reminder",
    "appointment_time": "2025-10-01T10:00:00"
  }
}
```

**Response:**
```json
{
  "call_id": "uuid-...",
  "call_sid": "CAxxxxxxxxxxxxxxxx",
  "status": "initiated",
  "to_number": "+91XXXXXXXXXX"
}
```

---

### `GET /calls/{call_id}`

Returns call record with transcript and summary.

---

### `POST /webhooks/twilio/status`

Twilio status callback webhook (auto-called by Twilio).

---

### `GET /webhooks/twilio/twiml`

Returns TwiML to initiate the Media Stream. Called by Twilio when the outbound call is answered.

---

### `WebSocket /ws/media-stream/{call_sid}`

Bidirectional audio stream between Twilio and the agent. Handles:
- `connected` — stream session established
- `start` — stream parameters (encoding, sample rate)
- `media` — base64-encoded μ-law audio chunks
- `stop` — stream termination

---

### `GET /health`

Returns `{"status": "ok"}`.

---

## Agent Node Reference

### `greeting_node`

- Sets `call_state = "GREETING"`
- Constructs the opening message using `AGENT_NAME` and call `context.purpose`
- Speaks the greeting via Deepgram TTS (synthesised to 8 kHz μ-law)
- Waits for caller's first utterance
- Transitions to `memory_retrieval`

### `memory_retrieval_node`

- Embeds the latest caller utterance locally with `all-MiniLM-L6-v2`
- Runs pgvector cosine similarity query (`TOP_K = MEMORY_RETRIEVAL_TOP_K`)
- Appends retrieved chunks to `AgentState.retrieved_memories`
- Transitions to `query_handler`

### `query_handler_node`

- Extracts `caller_intent` using a structured extraction prompt
- Detects `caller_sentiment` (positive / neutral / negative)
- Sets `should_escalate = True` if any `ESCALATION_KEYWORDS` are detected
- Transitions to `response_gen`

### `response_gen_node`

- Builds a context-enriched prompt:
  - System prompt from file
  - Retrieved memory chunks
  - Full conversation history
  - Current caller intent
- Composes the reply with Groq (`openai/gpt-oss-120b`); the reply is synthesised and streamed back via Deepgram TTS
- Appends agent turn to `conversation_history`
- Routes via `route_after_response()` conditional edge

### `escalation_node`

- Sets `call_state = "ESCALATION"`
- Plays a polite hold message to the caller
- Marks call record with `status = "escalated"` in DB
- Notifies human agent via configurable webhook (Slack, email, etc.)
- Transitions to `closing`

### `closing_node`

- Sets `call_state = "CLOSING"`
- Generates a concise call summary via a final LLM call
- Persists `summary`, `duration`, `status = "completed"` to `calls` table
- Persists full transcript to `transcripts` table
- Optionally upserts the resolved interaction as a new memory chunk
- Sends Twilio hangup command
- Returns `END`

---

## Call Flow Walkthrough

Below is a representative call with state transitions annotated:

```
[GREETING]
Agent: "Hi, this is Alex calling on behalf of Acme Corp.
        Am I speaking with Priya Sharma?"

[QUERYING] ← memory_retrieval fetches: appointment details, Priya's past interactions
Caller: "Yes, speaking. What is this about?"

[RESPONDING]
Agent: "Great! I'm reaching out to remind you about your appointment
        scheduled for October 1st at 10 AM. Does that still work for you?"

[QUERYING] ← memory_retrieval fetches: rescheduling policy
Caller: "Actually, I need to reschedule. Can I move it to the 3rd?"

[RESPONDING]
Agent: "Of course! Let me check availability for October 3rd...
        We have slots at 11 AM and 3 PM. Which works better?"

Caller: "3 PM works."

[RESPONDING]
Agent: "Perfect, I've noted October 3rd at 3 PM for you.
        Is there anything else I can help you with?"

Caller: "No, that's all."

[CLOSING]
Agent: "Wonderful! Have a great day, Priya. Goodbye!"
── call ends, summary persisted to DB ──
```

---

## Memory & RAG Pipeline

### Seeding the knowledge base

Prepare a JSONL file where each line is:

```json
{"content": "Our cancellation policy allows free rescheduling up to 24 hours before the appointment.", "source": "policy_doc", "metadata": {"category": "policy"}}
```

Run:

```bash
python scripts/seed_memory.py --input data/knowledge_base.jsonl
```

### Retrieval at runtime

```python
# app/memory/store.py (simplified)

async def retrieve_relevant_context(
    query_text: str,
    top_k: int = 5
) -> list[str]:
    embedding = await generate_embedding(query_text)
    
    result = await db.execute(
        """
        SELECT content
        FROM knowledge_base
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        [embedding, top_k]
    )
    
    return [row["content"] for row in result]
```

### Persisting resolved interactions

After a successful call, the closing node optionally upserts a summary as a new memory chunk — enabling the agent to learn from past conversations:

```python
await memory_store.upsert(
    content=f"Caller Priya Sharma rescheduled appointment to Oct 3 at 3PM.",
    source="call_resolution",
    metadata={"call_sid": call_sid, "outcome": "resolved"}
)
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test module
pytest tests/test_agent_graph.py -v

# Run integration tests only (requires running DB)
pytest -m integration
```

### Key test areas

| Test File | Coverage |
|---|---|
| `test_agent_graph.py` | LangGraph node execution, state transitions, conditional routing |
| `test_memory_store.py` | Embedding generation, pgvector upsert and retrieval |
| `test_twilio_webhook.py` | TwiML generation, Media Stream WebSocket protocol |
| `test_deepgram_client.py` | Deepgram STT/TTS clients, message parsing, audio framing |
| `test_media_stream.py` | Live-call orchestration (greeting, turns, escalation, closing) |

---

## Deployment

### Production Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: calling_agent
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./app/db/migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### Cloud deployment notes

- **Twilio webhooks** require a public HTTPS URL. Use a load balancer or API Gateway in front of your service.
- **WebSocket connections** for Media Streams must remain open for the duration of the call. Ensure your reverse proxy (nginx, ALB) is configured with appropriate timeout values (`proxy_read_timeout 3600s`).
- **pgvector** is supported natively on [Supabase](https://supabase.com), [Neon](https://neon.tech), and [AWS RDS for PostgreSQL 15+](https://aws.amazon.com/rds/).
- Ensure your server region is geographically close to Twilio's media servers for lowest audio latency.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Twilio webhook returns 403 | Signature validation failure | Ensure `TWILIO_AUTH_TOKEN` is correct; validate webhook URL matches exactly |
| Audio choppy / high latency | Network jitter or slow LLM response | Reduce `max_tokens` in Realtime session; check server proximity to Twilio |
| pgvector query slow | Missing HNSW index | Re-run `001_initial.sql`; verify index with `\d knowledge_base` |
| `WebSocket connection closed` | Twilio idle timeout | Send keep-alive media events every 30s |
| Agent stuck in QUERYING | Intent extraction failure | Check system prompt; add fallback intent `"unknown"` |
| Deepgram STT WebSocket auth failure | Wrong key | Verify `DEEPGRAM_API_KEY` (sent as `Authorization: Token ...`) |
| TTS returns 4xx/5xx | Model/params unsupported or bad key | Check `DEEPGRAM_TTS_MODEL` and that the key has TTS scope |
| Slow first embedding call | MiniLM model downloading on first use | Pre-warm with `python scripts/seed_memory.py` before taking calls |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Follow the existing code style (Black + isort + Ruff)
4. Add tests for new nodes or memory operations
5. Open a pull request with a clear description of changes

```bash
# Lint and format before pushing
black app/ tests/
isort app/ tests/
ruff check app/ tests/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using LangGraph · Groq · Realtime Voice API · Twilio · pgvector
</p>
