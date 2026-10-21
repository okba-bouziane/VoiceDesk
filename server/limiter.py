"""Call caps for the public demo (no framework imports: also used by the Vercel function)."""

from __future__ import annotations

import time


class SessionLimiter:
    """Protects the free provider quotas of a public demo: a daily cap on calls and a per-IP
    hourly cap. In memory only, so counts reset when the process restarts."""

    def __init__(self, daily_cap: int, per_ip_hour: int) -> None:
        self.daily_cap, self.per_ip_hour = daily_cap, per_ip_hour
        self._day = ""
        self._count = 0
        self._ip: dict[str, list[float]] = {}

    def check(self, ip: str, now: float | None = None) -> str | None:
        """Returns a refusal reason, or None when the call may start (and counts it)."""
        now = now or time.time()
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        if today != self._day:
            self._day, self._count = today, 0
        if self.daily_cap and self._count >= self.daily_cap:
            return "The daily demo limit has been reached. Please try again tomorrow."
        recent = [t for t in self._ip.get(ip, []) if now - t < 3600]
        if self.per_ip_hour and len(recent) >= self.per_ip_hour:
            return "You have started several calls already. Please try again in an hour."
        self._ip[ip] = [*recent, now]
        self._count += 1
        return None
