"""SQLite storage with the same rules as the Postgres store. Used by the hosted demo, where
there is no Postgres: `BOOKING_STORE=sqlite`. The gRPC contract is unchanged."""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
import uuid
from collections.abc import Callable
from typing import Any

from voicedesk.db import SAMPLE_PROVIDERS

from . import rules
from . import validation as v
from .models import AppointmentRow, SlotRow, clinic_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (id TEXT PRIMARY KEY, name TEXT NOT NULL,
    specialty TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS appointments (
    id TEXT PRIMARY KEY, patient_name TEXT NOT NULL, patient_phone TEXT NOT NULL,
    provider_id TEXT NOT NULL REFERENCES providers(id), starts_at TEXT NOT NULL,
    duration_min INTEGER NOT NULL DEFAULT 30, reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'booked');
CREATE UNIQUE INDEX IF NOT EXISTS one_live_slot
    ON appointments (provider_id, starts_at) WHERE status = 'booked';
"""

_SELECT_APPOINTMENT = (
    "SELECT a.*, p.name AS provider_name, p.specialty FROM appointments a "
    "JOIN providers p ON p.id = a.provider_id "
)


def _appointment(r: sqlite3.Row) -> AppointmentRow:
    start = dt.datetime.fromisoformat(r["starts_at"])
    return AppointmentRow(
        r["id"],
        r["patient_name"],
        r["patient_phone"],
        SlotRow(
            r["provider_id"],
            r["provider_name"],
            r["specialty"],
            start.date().isoformat(),
            start.strftime("%H:%M"),
            r["duration_min"],
        ),
        r["reason"],
        r["status"],
    )


class SqliteBookingStore:
    """One connection guarded by a lock: the demo traffic is tiny."""

    def __init__(self, path: str, clock: Callable[[], dt.datetime] = clinic_now) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.executemany(
                "INSERT OR IGNORE INTO providers VALUES (?, ?, ?)", SAMPLE_PROVIDERS
            )
            self._db.commit()

    def close(self) -> None:
        self._db.close()

    def _providers(self, specialty: str, provider_id: str) -> list[dict[str, Any]]:
        if provider_id:
            rows = self._db.execute(
                "SELECT * FROM providers WHERE id = ?", (provider_id,)
            ).fetchall()
            if not rows:
                raise rules.unknown_provider(provider_id)
            return [dict(r) for r in rows]
        if not specialty.strip():
            raise v.invalid("specialty or provider_id is required")
        rows = self._db.execute(
            "SELECT * FROM providers WHERE lower(specialty) = lower(?) ORDER BY id",
            (specialty.strip(),),
        ).fetchall()
        if not rows:
            known = self._db.execute("SELECT DISTINCT specialty FROM providers").fetchall()
            raise rules.unknown_specialty(specialty, (r["specialty"] for r in known))
        return [dict(r) for r in rows]

    def _get(self, appointment_id: str) -> AppointmentRow:
        row = self._db.execute(_SELECT_APPOINTMENT + "WHERE a.id = ?", (appointment_id,)).fetchone()
        if row is None:
            raise rules.appointment_not_found()
        return _appointment(row)

    def check_availability(
        self, specialty: str, provider_id: str, date: str, limit: int
    ) -> list[SlotRow]:
        now = self._clock()
        day, limit = rules.validate_availability_query(date, limit, now)
        with self._lock:
            providers = self._providers(specialty, provider_id)
            booked = self._db.execute(
                "SELECT provider_id, starts_at FROM appointments "
                "WHERE status = 'booked' AND starts_at LIKE ?",
                (day.isoformat() + "%",),
            )
            taken = {(r["provider_id"], r["starts_at"][11:16]) for r in booked}
        return rules.free_slots(providers, day, taken, now, limit)

    def book(
        self,
        name: str,
        phone: str,
        provider_id: str,
        specialty: str,
        date: str,
        start_time: str,
        reason: str,
    ) -> AppointmentRow:
        request = rules.validate_booking(name, phone, date, start_time, self._clock())
        with self._lock:
            for provider in self._providers(specialty, provider_id):
                aid = str(uuid.uuid4())
                try:
                    self._db.execute(
                        "INSERT INTO appointments (id, patient_name, patient_phone, provider_id, "
                        "starts_at, reason) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            aid,
                            request.name,
                            request.phone,
                            provider["id"],
                            request.starts_at.isoformat(timespec="minutes"),
                            (reason or "").strip()[:200],
                        ),
                    )
                    self._db.commit()
                except sqlite3.IntegrityError:
                    continue
                return self._get(aid)
        raise rules.slot_unavailable(request)

    def cancel(self, appointment_id: str, phone: str) -> AppointmentRow:
        aid = str(v.parse_uuid(appointment_id))
        phone = v.normalise_phone(phone)
        with self._lock:
            rules.check_cancellable(self._get(aid), phone, self._clock())
            self._db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (aid,))
            self._db.commit()
            return self._get(aid)

    def list_appointments(self, phone: str, include_cancelled: bool) -> list[AppointmentRow]:
        phone = v.normalise_phone(phone)
        sql = _SELECT_APPOINTMENT + "WHERE a.patient_phone = ? "
        if not include_cancelled:
            sql += "AND a.status = 'booked' "
        with self._lock:
            rows = self._db.execute(sql + "ORDER BY a.starts_at", (phone,)).fetchall()
        return [_appointment(r) for r in rows]
