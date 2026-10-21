"""SQLite storage with the same rules as the Postgres store. Used by the hosted demo, where
there is no Postgres: `BOOKING_STORE=sqlite`. The gRPC contract is unchanged."""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
import uuid
from collections.abc import Callable

import grpc

from voicedesk.db import SAMPLE_PROVIDERS

from . import validation as v
from .store import DEFAULT_LIMIT, MAX_LIMIT, AppointmentRow, SlotRow, clinic_now
from .validation import ServiceError


class SqliteBookingStore:
    def __init__(self, path: str, clock: Callable[[], dt.datetime] = clinic_now) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.executescript(
                """
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
            )
            self._db.executemany(
                "INSERT OR IGNORE INTO providers VALUES (?, ?, ?)", SAMPLE_PROVIDERS
            )
            self._db.commit()

    def close(self) -> None:
        self._db.close()

    def ping(self) -> bool:
        with self._lock:
            self._db.execute("SELECT 1")
        return True

    def _providers(self, specialty: str, provider_id: str) -> list[sqlite3.Row]:
        if provider_id:
            rows = self._db.execute(
                "SELECT * FROM providers WHERE id = ?", (provider_id,)
            ).fetchall()
            if not rows:
                raise ServiceError(grpc.StatusCode.NOT_FOUND, f"unknown provider '{provider_id}'")
            return rows
        if not specialty.strip():
            raise v.invalid("specialty or provider_id is required")
        rows = self._db.execute(
            "SELECT * FROM providers WHERE lower(specialty) = lower(?) ORDER BY id",
            (specialty.strip(),),
        ).fetchall()
        if not rows:
            known = [r[0] for r in self._db.execute("SELECT DISTINCT specialty FROM providers")]
            raise ServiceError(
                grpc.StatusCode.NOT_FOUND,
                f"no provider for specialty '{specialty}'. Available: {', '.join(sorted(known))}",
            )
        return rows

    def _get(self, appointment_id: str) -> AppointmentRow:
        r = self._db.execute(
            "SELECT a.*, p.name AS provider_name, p.specialty FROM appointments a "
            "JOIN providers p ON p.id = a.provider_id WHERE a.id = ?",
            (appointment_id,),
        ).fetchone()
        if r is None:
            raise ServiceError(grpc.StatusCode.NOT_FOUND, "appointment not found")
        return self._row(r)

    @staticmethod
    def _row(r: sqlite3.Row) -> AppointmentRow:
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

    def check_availability(
        self, specialty: str, provider_id: str, date: str, limit: int
    ) -> list[SlotRow]:
        day = v.parse_date(date)
        now = self._clock()
        v.check_bookable_day(day, now)
        limit = min(limit or DEFAULT_LIMIT, MAX_LIMIT)
        if limit < 0:
            raise v.invalid("limit must not be negative")
        with self._lock:
            providers = self._providers(specialty, provider_id)
            taken = {
                (r["provider_id"], r["starts_at"][11:16])
                for r in self._db.execute(
                    "SELECT provider_id, starts_at FROM appointments "
                    "WHERE status = 'booked' AND starts_at LIKE ?",
                    (day.isoformat() + "%",),
                )
            }
        free: list[SlotRow] = []
        for t in v.day_slots(day):
            if dt.datetime.combine(day, t) <= now:
                continue
            for p in providers:
                if (p["id"], t.strftime("%H:%M")) not in taken:
                    free.append(
                        SlotRow(
                            p["id"], p["name"], p["specialty"], day.isoformat(), t.strftime("%H:%M")
                        )
                    )
        return free[:limit]

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
        name = v.validate_name(name)
        phone = v.normalise_phone(phone)
        day = v.parse_date(date)
        t = v.parse_time(start_time)
        now = self._clock()
        v.check_bookable_day(day, now)
        v.check_slot_time(t)
        starts = dt.datetime.combine(day, t)
        if starts <= now:
            raise v.invalid("that time has already passed")
        with self._lock:
            for p in self._providers(specialty, provider_id):
                aid = str(uuid.uuid4())
                try:
                    self._db.execute(
                        "INSERT INTO appointments (id, patient_name, patient_phone, provider_id, "
                        "starts_at, reason) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            aid,
                            name,
                            phone,
                            p["id"],
                            starts.isoformat(timespec="minutes"),
                            (reason or "").strip()[:200],
                        ),
                    )
                    self._db.commit()
                except sqlite3.IntegrityError:
                    continue
                return self._get(aid)
        raise ServiceError(
            grpc.StatusCode.ALREADY_EXISTS,
            f"{day.isoformat()} {t.strftime('%H:%M')} is no longer available",
        )

    def cancel(self, appointment_id: str, phone: str) -> AppointmentRow:
        aid = str(v.parse_uuid(appointment_id))
        phone = v.normalise_phone(phone)
        with self._lock:
            appt = self._get(aid)
            if appt.patient_phone != phone:
                raise ServiceError(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "phone number does not match this appointment",
                )
            if appt.status == "cancelled":
                raise ServiceError(
                    grpc.StatusCode.FAILED_PRECONDITION, "appointment is already cancelled"
                )
            starts = dt.datetime.fromisoformat(f"{appt.slot.date}T{appt.slot.start_time}")
            if starts <= self._clock():
                raise ServiceError(
                    grpc.StatusCode.FAILED_PRECONDITION, "past appointments cannot be cancelled"
                )
            self._db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (aid,))
            self._db.commit()
            return self._get(aid)

    def list_appointments(self, phone: str, include_cancelled: bool) -> list[AppointmentRow]:
        phone = v.normalise_phone(phone)
        sql = (
            "SELECT a.*, p.name AS provider_name, p.specialty FROM appointments a "
            "JOIN providers p ON p.id = a.provider_id WHERE a.patient_phone = ? "
        )
        if not include_cancelled:
            sql += "AND a.status = 'booked' "
        with self._lock:
            return [self._row(r) for r in self._db.execute(sql + "ORDER BY a.starts_at", (phone,))]
