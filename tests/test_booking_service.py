"""gRPC booking service tests against an in-process server and a real Postgres."""

from __future__ import annotations

import datetime as dt
import uuid

import grpc
import psycopg
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

from booking_service.server import build_server
from booking_service.sqlite_store import SqliteBookingStore
from booking_service.store import BookingStore
from voicedesk.booking.v1 import booking_pb2 as pb
from voicedesk.booking.v1 import booking_pb2_grpc as pb_grpc

NOW = dt.datetime(2026, 10, 21, 10, 0)  # Wednesday 10:00
SUN, MON, FRI = "2026-10-25", "2026-10-26", "2026-10-23"
PHONE = "0555123456"


@pytest.fixture(params=["postgres", "sqlite"])
def stub(request, tmp_path):
    """The same contract tests run against both storage backends."""
    if request.param == "postgres":
        dsn = request.getfixturevalue("dsn")
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute("TRUNCATE appointments")
        store = BookingStore(dsn, clock=lambda: NOW)
    else:
        store = SqliteBookingStore(str(tmp_path / "t.db"), clock=lambda: NOW)
    server, port = build_server(store, "127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield pb_grpc.BookingServiceStub(channel), channel
    channel.close()
    server.stop(0)
    store.close()


def book(s, **kw):
    req = dict(
        patient_name="Sample Patient",
        patient_phone=PHONE,
        specialty="dentistry",
        date=SUN,
        start_time="10:00",
        reason="check-up",
    )
    req.update(kw)
    return s.BookAppointment(pb.BookAppointmentRequest(**req))


def code_of(call, *args):
    with pytest.raises(grpc.RpcError) as err:
        call(*args)
    return err.value.code()


def test_availability_lists_free_slots(stub):
    s, _ = stub
    r = s.CheckAvailability(pb.CheckAvailabilityRequest(specialty="dentistry", date=SUN, limit=4))
    assert [x.start_time for x in r.slots] == ["09:00", "09:30", "10:00", "10:30"]
    assert r.slots[0].provider_name == "Dr. Chaoui"


def test_availability_skips_lunch_and_two_providers_share_specialty(stub):
    s, _ = stub
    r = s.CheckAvailability(
        pb.CheckAvailabilityRequest(specialty="general medicine", date=SUN, limit=50)
    )
    times = {x.start_time for x in r.slots}
    assert "12:00" not in times and "12:30" not in times and "13:00" in times
    assert len(r.slots) == 2 * len(times)  # two general practitioners


def test_availability_hides_past_slots_today(stub):
    s, _ = stub
    r = s.CheckAvailability(pb.CheckAvailabilityRequest(specialty="dentistry", date="2026-10-21"))
    assert r.slots[0].start_time == "10:30"


def test_book_then_slot_is_taken(stub):
    s, _ = stub
    appt = book(s).appointment
    assert appt.status == pb.APPOINTMENT_STATUS_BOOKED and appt.slot.provider_id == "dr-chaoui"
    uuid.UUID(appt.appointment_id)
    r = s.CheckAvailability(pb.CheckAvailabilityRequest(specialty="dentistry", date=SUN, limit=3))
    assert "10:00" not in [x.start_time for x in r.slots]
    assert code_of(book, s) == grpc.StatusCode.ALREADY_EXISTS


def test_general_medicine_falls_back_to_second_provider(stub):
    s, _ = stub
    a = book(s, specialty="general medicine").appointment
    b = book(s, specialty="general medicine", patient_name="Other Patient").appointment
    assert {a.slot.provider_id, b.slot.provider_id} == {"dr-amrani", "dr-benali"}
    assert code_of(lambda: book(s, specialty="general medicine")) == grpc.StatusCode.ALREADY_EXISTS


@pytest.mark.parametrize(
    "kw",
    [
        {"date": "not-a-date"},
        {"date": "2026-10-20"},  # past
        {"date": FRI},  # clinic closed
        {"date": "2027-06-01"},  # too far ahead
        {"start_time": "10:15"},  # off the 30 minute grid
        {"start_time": "12:30"},  # lunch break
        {"start_time": "16:30"},  # after the last slot
        {"start_time": "9am"},
        {"patient_phone": "12"},
        {"patient_name": " "},
        {"specialty": ""},
    ],
)
def test_book_validation_returns_invalid_argument(stub, kw):
    s, _ = stub
    assert code_of(lambda: book(s, **kw)) == grpc.StatusCode.INVALID_ARGUMENT


def test_unknown_specialty_is_not_found(stub):
    s, _ = stub
    assert code_of(lambda: book(s, specialty="astrology")) == grpc.StatusCode.NOT_FOUND


def test_cancel_flow_and_status_codes(stub):
    s, _ = stub
    appt = book(s).appointment
    cancel = lambda **kw: s.CancelAppointment(pb.CancelAppointmentRequest(**kw))  # noqa: E731
    assert code_of(lambda: cancel(appointment_id="nope", patient_phone=PHONE)) == (
        grpc.StatusCode.INVALID_ARGUMENT
    )
    assert code_of(lambda: cancel(appointment_id=str(uuid.uuid4()), patient_phone=PHONE)) == (
        grpc.StatusCode.NOT_FOUND
    )
    assert code_of(
        lambda: cancel(appointment_id=appt.appointment_id, patient_phone="0777000000")
    ) == (grpc.StatusCode.PERMISSION_DENIED)
    done = cancel(appointment_id=appt.appointment_id, patient_phone=PHONE).appointment
    assert done.status == pb.APPOINTMENT_STATUS_CANCELLED
    assert code_of(lambda: cancel(appointment_id=appt.appointment_id, patient_phone=PHONE)) == (
        grpc.StatusCode.FAILED_PRECONDITION
    )
    # the slot is free again
    assert book(s).appointment.slot.start_time == "10:00"


def test_list_appointments_sorted_and_filters_cancelled(stub):
    s, _ = stub
    late = book(s, start_time="11:00").appointment
    early = book(s, start_time="09:00").appointment
    s.CancelAppointment(
        pb.CancelAppointmentRequest(appointment_id=late.appointment_id, patient_phone=PHONE)
    )
    live = s.ListAppointments(pb.ListAppointmentsRequest(patient_phone=PHONE)).appointments
    assert [a.appointment_id for a in live] == [early.appointment_id]
    every = s.ListAppointments(
        pb.ListAppointmentsRequest(patient_phone=PHONE, include_cancelled=True)
    ).appointments
    assert len(every) == 2
    assert not s.ListAppointments(
        pb.ListAppointmentsRequest(patient_phone="0600000000")
    ).appointments


def test_phone_is_normalised(stub):
    s, _ = stub
    a = book(s, patient_phone="0555 12 34 56").appointment
    assert a.patient_phone == PHONE


def test_health_check_and_reflection(stub):
    _, channel = stub
    h = health_pb2_grpc.HealthStub(channel)
    assert h.Check(health_pb2.HealthCheckRequest()).status == health_pb2.HealthCheckResponse.SERVING
    named = h.Check(health_pb2.HealthCheckRequest(service="voicedesk.booking.v1.BookingService"))
    assert named.status == health_pb2.HealthCheckResponse.SERVING

    refl = reflection_pb2_grpc.ServerReflectionStub(channel)
    responses = refl.ServerReflectionInfo(
        iter([reflection_pb2.ServerReflectionRequest(list_services="")])
    )
    names = {s.name for r in responses for s in r.list_services_response.service}
    assert "voicedesk.booking.v1.BookingService" in names
