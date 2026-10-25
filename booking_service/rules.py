"""Business rules shared by every storage backend, so Postgres and SQLite cannot drift apart."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import grpc

from . import validation as v
from .models import AppointmentRow, SlotRow
from .validation import ServiceError

DEFAULT_LIMIT = 10
MAX_LIMIT = 50


@dataclass(frozen=True)
class BookingRequest:
    """A booking request after validation."""

    name: str
    phone: str
    day: dt.date
    time: dt.time

    @property
    def starts_at(self) -> dt.datetime:
        return dt.datetime.combine(self.day, self.time)


def validate_booking(
    name: str, phone: str, date: str, start_time: str, now: dt.datetime
) -> BookingRequest:
    request = BookingRequest(
        name=v.validate_name(name),
        phone=v.normalise_phone(phone),
        day=v.parse_date(date),
        time=v.parse_time(start_time),
    )
    v.check_bookable_day(request.day, now)
    v.check_slot_time(request.time)
    if request.starts_at <= now:
        raise v.invalid("that time has already passed")
    return request


def validate_availability_query(date: str, limit: int, now: dt.datetime) -> tuple[dt.date, int]:
    day = v.parse_date(date)
    v.check_bookable_day(day, now)
    if limit < 0:
        raise v.invalid("limit must not be negative")
    return day, min(limit or DEFAULT_LIMIT, MAX_LIMIT)


def free_slots(
    providers: Iterable[Mapping[str, Any]],
    day: dt.date,
    taken: set[tuple[str, str]],
    now: dt.datetime,
    limit: int,
) -> list[SlotRow]:
    """Open slots of `day` in time order, skipping the past and `taken` (provider_id, HH:MM)."""
    providers = list(providers)
    free: list[SlotRow] = []
    for start in v.day_slots(day):
        if dt.datetime.combine(day, start) <= now:
            continue
        hhmm = start.strftime("%H:%M")
        free.extend(
            SlotRow(p["id"], p["name"], p["specialty"], day.isoformat(), hhmm)
            for p in providers
            if (p["id"], hhmm) not in taken
        )
    return free[:limit]


def check_cancellable(appt: AppointmentRow, phone: str, now: dt.datetime) -> None:
    if appt.patient_phone != phone:
        raise ServiceError(
            grpc.StatusCode.PERMISSION_DENIED, "phone number does not match this appointment"
        )
    if appt.status == "cancelled":
        raise ServiceError(grpc.StatusCode.FAILED_PRECONDITION, "appointment is already cancelled")
    if dt.datetime.fromisoformat(f"{appt.slot.date}T{appt.slot.start_time}") <= now:
        raise ServiceError(
            grpc.StatusCode.FAILED_PRECONDITION, "past appointments cannot be cancelled"
        )


def unknown_provider(provider_id: str) -> ServiceError:
    return ServiceError(grpc.StatusCode.NOT_FOUND, f"unknown provider '{provider_id}'")


def unknown_specialty(specialty: str, known: Iterable[str]) -> ServiceError:
    return ServiceError(
        grpc.StatusCode.NOT_FOUND,
        f"no provider for specialty '{specialty}'. Available: {', '.join(sorted(known))}",
    )


def appointment_not_found() -> ServiceError:
    return ServiceError(grpc.StatusCode.NOT_FOUND, "appointment not found")


def slot_unavailable(request: BookingRequest) -> ServiceError:
    return ServiceError(
        grpc.StatusCode.ALREADY_EXISTS,
        f"{request.day.isoformat()} {request.time.strftime('%H:%M')} is no longer available",
    )
