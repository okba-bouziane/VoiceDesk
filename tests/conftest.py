from __future__ import annotations

import os

import psycopg
import pytest

from voicedesk.config import get_settings
from voicedesk.db import init_schema

TEST_DB = "voicedesk_test"


def _admin_dsn() -> str:
    return get_settings().database_url.rsplit("/", 1)[0] + "/postgres"


@pytest.fixture(scope="session")
def dsn() -> str:
    """A dedicated test database (created on demand) with schema and sample providers."""
    try:
        with psycopg.connect(_admin_dsn(), autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                conn.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError as err:
        if os.getenv("CI") or os.getenv("REQUIRE_DB"):
            raise
        pytest.skip(f"Postgres not reachable: {err}")
    test_dsn = get_settings().database_url.rsplit("/", 1)[0] + f"/{TEST_DB}"
    init_schema(test_dsn)
    return test_dsn
