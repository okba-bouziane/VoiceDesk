"""Pure unit tests of the booking rules and helpers shared by both storage backends."""

from __future__ import annotations

import datetime as dt

import grpc
import pytest

from agent.demo_guard import daily_cap_reached
from booking_service import rules
from booking_service.models import AppointmentRow, SlotRow
from booking_service.validation import ServiceError
from voicedesk.phone import is_valid_phone, normalise_phone

NOW = dt.datetime(2026, 10, 21, 10, 0)  # Wednesday 10:00
PROVIDERS = [{"id": "dr-a", "name": "Dr. A", "specialty": "dentistry"}]


def test_phone_normalisation_and_validation():
    assert normalise_phone("+213 (555) 12-34.56") == "+213555123456"
    assert is_valid_phone("0555123456")
    assert not is_valid_phone("12345") and not is_valid_phone("05551234ab")


def test_free_slots_skips_past_taken_and_lunch():
    taken = {("dr-a", "10:30")}
    slots = rules.free_slots(PROVIDERS, NOW.date(), taken, NOW, limit=50)
    times = [s.start_time for s in slots]
    assert times[0] == "11:00"  # 10:00 is not after now, 10:30 is taken
    assert "12:00" not in times and "12:30" not in times  # lunch break
    assert rules.free_slots(PROVIDERS, NOW.date(), taken, NOW, limit=2) == slots[:2]


def test_availability_query_limits():
    assert rules.validate_availability_query("2026-10-25", 0, NOW)[1] == rules.DEFAULT_LIMIT
    assert rules.validate_availability_query("2026-10-25", 999, NOW)[1] == rules.MAX_LIMIT
    with pytest.raises(ServiceError) as err:
        rules.validate_availability_query("2026-10-25", -1, NOW)
    assert err.value.code is grpc.StatusCode.INVALID_ARGUMENT


def _appointment(status: str = "booked", start: str = "2026-10-25", time: str = "09:00"):
    return AppointmentRow(
        "id",
        "Sample Patient",
        "0555123456",
        SlotRow("dr-a", "Dr. A", "dentistry", start, time),
        "",
        status,
    )


@pytest.mark.parametrize(
    ("appointment", "phone", "status"),
    [
        (_appointment(), "0555000000", grpc.StatusCode.PERMISSION_DENIED),
        (_appointment(status="cancelled"), "0555123456", grpc.StatusCode.FAILED_PRECONDITION),
        (_appointment(start="2026-10-20"), "0555123456", grpc.StatusCode.FAILED_PRECONDITION),
    ],
)
def test_cancel_rules(appointment, phone, status):
    with pytest.raises(ServiceError) as err:
        rules.check_cancellable(appointment, phone, NOW)
    assert err.value.code is status


def test_cancel_allows_future_booked_appointment_of_the_caller():
    rules.check_cancellable(_appointment(), "0555123456", NOW)


def test_daily_cap_counts_calls_and_can_be_disabled(tmp_path):
    counter = str(tmp_path / "calls.json")
    assert not daily_cap_reached(0, counter)  # disabled: never refuses, never counts
    assert [daily_cap_reached(2, counter) for _ in range(3)] == [False, False, True]
