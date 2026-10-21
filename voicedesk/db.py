"""Postgres helpers: connection, schema bootstrap and sample provider data."""

from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

# Sample data: fictional providers.
SAMPLE_PROVIDERS = [
    ("dr-amrani", "Dr. Amrani", "general medicine"),
    ("dr-benali", "Dr. Benali", "general medicine"),
    ("dr-chaoui", "Dr. Chaoui", "dentistry"),
    ("dr-djerbi", "Dr. Djerbi", "pediatrics"),
    ("dr-ferhat", "Dr. Ferhat", "dermatology"),
]


def connect(dsn: str, *, vector: bool = False) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    if vector:
        register_vector(conn)
    return conn


def init_schema(dsn: str) -> None:
    """Create extension, tables and indexes, and seed the sample providers."""
    with connect(dsn) as conn:
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        for pid, name, specialty in SAMPLE_PROVIDERS:
            conn.execute(
                "INSERT INTO providers (id, name, specialty) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (pid, name, specialty),
            )
