"""Postgres-backed storage of the booking service (rules live in `rules.py`)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import rules
from . import validation as v
from .models import AppointmentRow, SlotRow, clinic_now

Row = dict[str, Any]

_SELECT_APPOINTMENT = (
    "SELECT a.*, p.name AS provider_name, p.specialty FROM appointments a "
    "JOIN providers p ON p.id = a.provider_id "
)


def _appointment(row: Row) -> AppointmentRow:
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


class BookingStore:
    def __init__(self, dsn: str, clock: Callable[[], dt.datetime] = clinic_now) -> None:
        self._pool: ConnectionPool[Any] = ConnectionPool(
            dsn, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, open=True
        )
        self._clock = clock

    def close(self) -> None:
        self._pool.close()

    def _providers(
        self, conn: psycopg.Connection[Any], specialty: str, provider_id: str
    ) -> list[Row]:
        if provider_id:
            rows = conn.execute("SELECT * FROM providers WHERE id = %s", (provider_id,)).fetchall()
            if not rows:
                raise rules.unknown_provider(provider_id)
            return list(rows)
        if not specialty.strip():
            raise v.invalid("specialty or provider_id is required")
        rows = conn.execute(
            "SELECT * FROM providers WHERE lower(specialty) = lower(%s) ORDER BY id",
            (specialty.strip(),),
        ).fetchall()
        if not rows:
            known = conn.execute("SELECT DISTINCT specialty FROM providers").fetchall()
            raise rules.unknown_specialty(specialty, (r["specialty"] for r in known))
        return list(rows)

    def _get(self, conn: psycopg.Connection[Any], appointment_id: str) -> AppointmentRow:
        row = conn.execute(_SELECT_APPOINTMENT + "WHERE a.id = %s", (appointment_id,)).fetchone()
        if row is None:
            raise rules.appointment_not_found()
        return _appointment(row)

    def check_availability(
        self, specialty: str, provider_id: str, date: str, limit: int
    ) -> list[SlotRow]:
        now = self._clock()
        day, limit = rules.validate_availability_query(date, limit, now)
        with self._pool.connection() as conn:
            providers = self._providers(conn, specialty, provider_id)
            booked = conn.execute(
                "SELECT provider_id, starts_at FROM appointments "
                "WHERE status = 'booked' AND starts_at >= %s AND starts_at < %s",
                (day, day + dt.timedelta(days=1)),
            )
            taken = {(r["provider_id"], r["starts_at"].strftime("%H:%M")) for r in booked}
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
        with self._pool.connection() as conn:
            for provider in self._providers(conn, specialty, provider_id):
                try:
                    with conn.transaction():  # savepoint: a lost race must not poison the txn
                        row = conn.execute(
                            "INSERT INTO appointments (patient_name, patient_phone, provider_id, "
                            "starts_at, reason) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                            (
                                request.name,
                                request.phone,
                                provider["id"],
                                request.starts_at,
                                (reason or "").strip()[:200],
                            ),
                        ).fetchone()
                except UniqueViolation:
                    continue
                assert row is not None
                return self._get(conn, str(row["id"]))
        raise rules.slot_unavailable(request)

    def cancel(self, appointment_id: str, phone: str) -> AppointmentRow:
        aid = str(v.parse_uuid(appointment_id))
        phone = v.normalise_phone(phone)
        with self._pool.connection() as conn:
            rules.check_cancellable(self._get(conn, aid), phone, self._clock())
            conn.execute("UPDATE appointments SET status = 'cancelled' WHERE id = %s", (aid,))
            return self._get(conn, aid)

    def list_appointments(self, phone: str, include_cancelled: bool) -> list[AppointmentRow]:
        phone = v.normalise_phone(phone)
        sql = _SELECT_APPOINTMENT + "WHERE a.patient_phone = %s "
        if not include_cancelled:
            sql += "AND a.status = 'booked' "
        with self._pool.connection() as conn:
            rows = conn.execute(sql + "ORDER BY a.starts_at", (phone,)).fetchall()
        return [_appointment(r) for r in rows]
