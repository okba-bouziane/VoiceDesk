"""Guards that keep a public demo inside the free provider quotas."""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path


def daily_cap_reached(cap: int, counter_path: str) -> bool:
    """Count one call against today's (UTC) cap, shared by all job processes through a file.

    Returns True when the cap is already used up. A cap of 0 disables the guard."""
    if not cap:
        return False
    path = Path(counter_path)
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        if data.get("day") != today:
            data = {"day": today, "count": 0}
        if data["count"] >= cap:
            return True
        data["count"] += 1
        path.write_text(json.dumps(data))
    return False
