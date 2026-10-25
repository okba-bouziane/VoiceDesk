"""Create the schema, seed sample providers and (re)index the clinic documents."""

from __future__ import annotations

import logging

from voicedesk.config import get_settings
from voicedesk.db import init_schema_when_ready
from voicedesk.rag import ingest_documents


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dsn = get_settings().database_url
    init_schema_when_ready(dsn, attempts=20)
    print(f"indexed {ingest_documents(dsn)} chunks from data/clinic_docs")


if __name__ == "__main__":
    main()
