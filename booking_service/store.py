"""Postgres-backed storage and business rules of the booking service."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import grpc
import psycopg
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import validation as v
from .validation import ServiceError

CLINIC_TZ = ZoneInfo("Africa/Algiers")
DEFAULT_LIMIT = 10
MAX_LIMIT = 50


def clinic_now() -> dt.datetime:
    return dt.datetime.now(CLINIC_TZ).replace(tzinfo=None)


@dataclass
class SlotRow:
    provider_id: str
    provider_name: str
    specialty: str
    date: str
    start_time: str
    duration_minutes: int = v.SLOT_MINUTES


@dataclass
class AppointmentRow:
    appointment_id: str
    patient_name: str
    patient_phone: str
    slot: SlotRow
    reason: str
    status: str


class BookingStore:
    def __init__(self, dsn: str, clock: Callable[[], dt.datetime] = clinic_now) -> None:
        self._pool: ConnectionPool[Any] = ConnectionPool(
            dsn, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, open=True
        )
        self._clock = clock

    def close(self) -> None:
        self._pool.close()

    def ping(self) -> bool:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    # -- helpers ---------------------------------------------------------------------------
    def _providers(
        self, conn: psycopg.Connection[Any], specialty: str, provider_id: str
    ) -> list[dict[str, Any]]:
        if provider_id:
            rows = conn.execute("SELECT * FROM providers WHERE id = %s", (provider_id,)).fetchall()
            if not rows:
                raise ServiceError(grpc.StatusCode.NOT_FOUND, f"unknown provider '{provider_id}'")
            return list(rows)
        if not specialty.strip():
            raise v.invalid("specialty or provider_id is required")
        rows = conn.execute(
            "SELECT * FROM providers WHERE lower(specialty) = lower(%s) ORDER BY id",
            (specialty.strip(),),
        ).fetchall()
        if not rows:
            known = [
                r["specialty"]
                for r in conn.execute("SELECT DISTINCT specialty FROM providers ORDER BY 1")
            ]
            raise ServiceError(
                grpc.StatusCode.NOT_FOUND,
                f"no provider for specialty '{specialty}'. Available: {', '.join(known)}",
            )
        return list(rows)

    @staticmethod
    def _slot(provider: dict[str, Any], day: dt.date, t: dt.time) -> SlotRow:
        return SlotRow(
            provider["id"],
            provider["name"],
            provider["specialty"],
            day.isoformat(),
            t.strftime("%H:%M"),
        )

    def _appointment(self, row: dict[str, Any]) -> AppointmentRow:
        start: dt.datetime = row["starts_at"]
        return AppointmentRow(
            appointment_id=str(row["id"]),
            patient_name=row["patient_name"],
            patient_phone=row["patient_phone"],
            slot=SlotRow(
                row["provider_id"],
                row["provider_name"],
                row["specialty"],
                start.date().isoformat(),
                start.strftime("%H:%M"),
                row["duration_min"],
            ),
            reason=row["reason"],
            status=row["status"],
        )

    _SELECT = (
        "SELECT a.*, p.name AS provider_name, p.specialty FROM appointments a "
        "JOIN providers p ON p.id = a.provider_id "
    )

    # -- operations ------------------------------------------------------------------------
    def check_availability(
        self, specialty: str, provider_id: str, date: str, limit: int
    ) -> list[SlotRow]:
        day = v.parse_date(date)
        now = self._clock()
        v.check_bookable_day(day, now)
        limit = min(limit or DEFAULT_LIMIT, MAX_LIMIT)
        if limit < 0:
            raise v.invalid("limit must not be negative")
        with self._pool.connection() as conn:
            providers = self._providers(conn, specialty, provider_id)
            taken = {
                (r["provider_id"], r["starts_at"].time())
                for r in conn.execute(
                    "SELECT provider_id, starts_at FROM appointments "
                    "WHERE status = 'booked' AND starts_at >= %s AND starts_at < %s",
                    (day, day + dt.timedelta(days=1)),
                )
            }
        free: list[SlotRow] = []
        for t in v.day_slots(day):
            if dt.datetime.combine(day, t) <= now:
                continue
            for p in providers:
                if (p["id"], t) not in taken:
                    free.append(self._slot(p, day, t))
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
        with self._pool.connection() as conn:
            providers = self._providers(conn, specialty, provider_id)
            for p in providers:
                try:
                    with conn.transaction():  # savepoint: a lost race must not poison the txn
                        row = conn.execute(
                            "INSERT INTO appointments (patient_name, patient_phone, provider_id, "
                            "starts_at, reason) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                            (name, phone, p["id"], starts, (reason or "").strip()[:200]),
                        ).fetchone()
                except UniqueViolation:
                    continue
                assert row is not None
                return self._get(conn, str(row["id"]))
        raise ServiceError(
            grpc.StatusCode.ALREADY_EXISTS,
            f"{day.isoformat()} {t.strftime('%H:%M')} is no longer available",
        )

    def _get(self, conn: psycopg.Connection[Any], appointment_id: str) -> AppointmentRow:
        row = conn.execute(self._SELECT + "WHERE a.id = %s", (appointment_id,)).fetchone()
        if row is None:
            raise ServiceError(grpc.StatusCode.NOT_FOUND, "appointment not found")
        return self._appointment(row)

    def cancel(self, appointment_id: str, phone: str) -> AppointmentRow:
        aid = v.parse_uuid(appointment_id)
        phone = v.normalise_phone(phone)
        with self._pool.connection() as conn:
            appt = self._get(conn, str(aid))
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
            conn.execute("UPDATE appointments SET status = 'cancelled' WHERE id = %s", (str(aid),))
            return self._get(conn, str(aid))

    def list_appointments(self, phone: str, include_cancelled: bool) -> list[AppointmentRow]:
        phone = v.normalise_phone(phone)
        sql = self._SELECT + "WHERE a.patient_phone = %s "
        if not include_cancelled:
            sql += "AND a.status = 'booked' "
        sql += "ORDER BY a.starts_at"
        with self._pool.connection() as conn:
            return [self._appointment(r) for r in conn.execute(sql, (phone,)).fetchall()]
