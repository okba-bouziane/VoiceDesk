"""Create the schema, seed sample providers and (re)index the clinic documents."""

from __future__ import annotations

import time

from voicedesk.config import get_settings
from voicedesk.db import init_schema
from voicedesk.rag import ingest_documents


def main() -> None:
    dsn = get_settings().database_url
    for attempt in range(20):
        try:
            init_schema(dsn)
            break
        except Exception as err:  # noqa: BLE001
            print(f"database not ready ({err}); retry {attempt + 1}")
            time.sleep(2)
    else:
        raise SystemExit("database never became ready")
    n = ingest_documents(dsn)
    print(f"indexed {n} chunks from data/clinic_docs")


if __name__ == "__main__":
    main()
