"""Row types and the storage interface of the booking service."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

from voicedesk.config import CLINIC_TZ

from .validation import SLOT_MINUTES


def clinic_now() -> dt.datetime:
    """Current clinic-local time as a naive datetime (the database stores local times)."""
    return dt.datetime.now(CLINIC_TZ).replace(tzinfo=None)


@dataclass
class SlotRow:
    provider_id: str
    provider_name: str
    specialty: str
    date: str
    start_time: str
    duration_minutes: int = SLOT_MINUTES


@dataclass
class AppointmentRow:
    appointment_id: str
    patient_name: str
    patient_phone: str
    slot: SlotRow
    reason: str
    status: str


class Store(Protocol):
    """What the gRPC servicer needs from a storage backend (Postgres or SQLite)."""

    def close(self) -> None: ...

    def check_availability(
        self, specialty: str, provider_id: str, date: str, limit: int
    ) -> list[SlotRow]: ...

    def book(
        self,
        name: str,
        phone: str,
        provider_id: str,
        specialty: str,
        date: str,
        start_time: str,
        reason: str,
    ) -> AppointmentRow: ...

    def cancel(self, appointment_id: str, phone: str) -> AppointmentRow: ...

    def list_appointments(self, phone: str, include_cancelled: bool) -> list[AppointmentRow]: ...
