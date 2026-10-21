CREATE EXTENSION IF NOT EXISTS vector;

-- Sample data: fictional providers of a fictional clinic.
CREATE TABLE IF NOT EXISTS providers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    specialty   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_name   TEXT NOT NULL,
    patient_phone  TEXT NOT NULL,
    provider_id    TEXT NOT NULL REFERENCES providers(id),
    starts_at      TIMESTAMP NOT NULL,           -- clinic-local time
    duration_min   INTEGER NOT NULL DEFAULT 30,
    reason         TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'booked' CHECK (status IN ('booked', 'cancelled')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One live booking per provider and start time; cancelled rows free the slot.
CREATE UNIQUE INDEX IF NOT EXISTS appointments_one_live_slot
    ON appointments (provider_id, starts_at) WHERE status = 'booked';
CREATE INDEX IF NOT EXISTS appointments_phone_idx ON appointments (patient_phone);

CREATE TABLE IF NOT EXISTS doc_chunks (
    id         SERIAL PRIMARY KEY,
    source     TEXT NOT NULL,
    heading    TEXT NOT NULL,
    content    TEXT NOT NULL,
    embedding  vector(384) NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (
                   to_tsvector('english', heading || ' ' || content)
               ) STORED
);
CREATE INDEX IF NOT EXISTS doc_chunks_tsv_idx ON doc_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
    ON doc_chunks USING hnsw (embedding vector_cosine_ops);
