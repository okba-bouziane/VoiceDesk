"""Input validation for the booking service. Raises ServiceError with a gRPC status."""

from __future__ import annotations

import datetime as dt
import re
import uuid

import grpc

SLOT_MINUTES = 30
OPEN_DAYS = {6, 0, 1, 2, 3}  # Sunday to Thursday (date.weekday(): Monday=0)
OPEN_FROM = dt.time(9, 0)
OPEN_UNTIL = dt.time(16, 30)  # last slot ends here
LUNCH = (dt.time(12, 0), dt.time(13, 0))
MAX_DAYS_AHEAD = 90

_PHONE_RE = re.compile(r"^\+?\d{8,15}$")


class ServiceError(Exception):
    def __init__(self, code: grpc.StatusCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def invalid(message: str) -> ServiceError:
    return ServiceError(grpc.StatusCode.INVALID_ARGUMENT, message)


def parse_date(value: str) -> dt.date:
    if not value:
        raise invalid("date is required (YYYY-MM-DD)")
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise invalid(f"date '{value}' is not a valid YYYY-MM-DD date") from None


def parse_time(value: str) -> dt.time:
    if not value:
        raise invalid("start_time is required (HH:MM)")
    if not re.fullmatch(r"\d{2}:\d{2}", value):
        raise invalid(f"start_time '{value}' must look like HH:MM")
    try:
        return dt.time.fromisoformat(value)
    except ValueError:
        raise invalid(f"start_time '{value}' is not a valid time") from None


def normalise_phone(value: str) -> str:
    cleaned = re.sub(r"[\s\-().]", "", value or "")
    if not _PHONE_RE.match(cleaned):
        raise invalid("patient_phone must contain 8 to 15 digits, optionally starting with +")
    return cleaned


def validate_name(value: str) -> str:
    name = (value or "").strip()
    if not 2 <= len(name) <= 80:
        raise invalid("patient_name must be between 2 and 80 characters")
    return name


def parse_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise invalid("appointment_id is not a valid UUID") from None


def check_bookable_day(day: dt.date, now: dt.datetime) -> None:
    if day < now.date():
        raise invalid(f"{day.isoformat()} is in the past")
    if day > now.date() + dt.timedelta(days=MAX_DAYS_AHEAD):
        raise invalid(f"bookings open at most {MAX_DAYS_AHEAD} days ahead")
    if day.weekday() not in OPEN_DAYS:
        raise invalid(f"the clinic is closed on {day.strftime('%A')}s")


def check_slot_time(t: dt.time) -> None:
    last_start = dt.datetime.combine(dt.date.min, OPEN_UNTIL) - dt.timedelta(minutes=SLOT_MINUTES)
    if t.second or t.microsecond or t.minute % SLOT_MINUTES:
        raise invalid(f"start_time must be on a {SLOT_MINUTES}-minute boundary")
    if t < OPEN_FROM or t > last_start.time() or LUNCH[0] <= t < LUNCH[1]:
        raise invalid(
            f"start_time {t.strftime('%H:%M')} is outside bookable hours "
            "(09:00-12:00 and 13:00-16:00)"
        )


def day_slots(day: dt.date) -> list[dt.time]:
    """All slot start times of a day, before removing booked ones."""
    slots: list[dt.time] = []
    cur = dt.datetime.combine(day, OPEN_FROM)
    end = dt.datetime.combine(day, OPEN_UNTIL)
    while cur + dt.timedelta(minutes=SLOT_MINUTES) <= end:
        if not (LUNCH[0] <= cur.time() < LUNCH[1]):
            slots.append(cur.time())
        cur += dt.timedelta(minutes=SLOT_MINUTES)
    return slots
