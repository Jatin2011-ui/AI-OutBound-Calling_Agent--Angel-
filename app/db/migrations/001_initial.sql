-- 001_initial.sql — initial schema for the AI Outbound Calling Agent.
-- Requires PostgreSQL 15+.

CREATE EXTENSION IF NOT EXISTS vector;

-- Stores outbound call records
CREATE TABLE IF NOT EXISTS calls (
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
CREATE TABLE IF NOT EXISTS transcripts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id     UUID REFERENCES calls(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('agent', 'caller')),
    content     TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Semantic memory / knowledge base
CREATE TABLE IF NOT EXISTS knowledge_base (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    embedding   vector(384),                   -- pgvector column (all-MiniLM-L6-v2)
    source      TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast approximate nearest neighbour search (cosine)
CREATE INDEX IF NOT EXISTS knowledge_base_embedding_hnsw
    ON knowledge_base
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
