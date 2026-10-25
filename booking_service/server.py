"""gRPC server of the booking microservice (health check and reflection included)."""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from concurrent import futures
from typing import TypeVar

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from voicedesk.booking.v1 import booking_pb2 as pb
from voicedesk.booking.v1 import booking_pb2_grpc as pb_grpc
from voicedesk.config import Settings, get_settings
from voicedesk.db import init_schema_when_ready

from .models import AppointmentRow, SlotRow, Store
from .sqlite_store import SqliteBookingStore
from .store import BookingStore
from .validation import ServiceError

log = logging.getLogger("booking_service")
R = TypeVar("R")

_STATUS = {
    "booked": pb.APPOINTMENT_STATUS_BOOKED,
    "cancelled": pb.APPOINTMENT_STATUS_CANCELLED,
}


def _slot_pb(s: SlotRow) -> pb.Slot:
    return pb.Slot(
        provider_id=s.provider_id,
        provider_name=s.provider_name,
        specialty=s.specialty,
        date=s.date,
        start_time=s.start_time,
        duration_minutes=s.duration_minutes,
    )


def _appt_pb(a: AppointmentRow) -> pb.Appointment:
    return pb.Appointment(
        appointment_id=a.appointment_id,
        patient_name=a.patient_name,
        patient_phone=a.patient_phone,
        slot=_slot_pb(a.slot),
        reason=a.reason,
        status=_STATUS[a.status],
    )


class BookingServicer(pb_grpc.BookingServiceServicer):
    def __init__(self, store: Store) -> None:
        self._store = store

    @staticmethod
    def _guard(context: grpc.ServicerContext, handler: Callable[[], R]) -> R:
        """Run a handler and turn failures into gRPC statuses without leaking internals."""
        try:
            return handler()
        except ServiceError as err:
            context.abort(err.code, err.message)
        except Exception:
            log.exception("unhandled error")
            context.abort(grpc.StatusCode.INTERNAL, "internal error")
        raise AssertionError("unreachable: context.abort raises")

    def CheckAvailability(  # noqa: N802
        self, request: pb.CheckAvailabilityRequest, context: grpc.ServicerContext
    ) -> pb.CheckAvailabilityResponse:
        def run() -> pb.CheckAvailabilityResponse:
            slots = self._store.check_availability(
                request.specialty, request.provider_id, request.date, request.limit
            )
            return pb.CheckAvailabilityResponse(slots=[_slot_pb(s) for s in slots])

        return self._guard(context, run)

    def BookAppointment(  # noqa: N802
        self, request: pb.BookAppointmentRequest, context: grpc.ServicerContext
    ) -> pb.BookAppointmentResponse:
        def run() -> pb.BookAppointmentResponse:
            appt = self._store.book(
                request.patient_name,
                request.patient_phone,
                request.provider_id,
                request.specialty,
                request.date,
                request.start_time,
                request.reason,
            )
            log.info("booked %s", appt.appointment_id)
            return pb.BookAppointmentResponse(appointment=_appt_pb(appt))

        return self._guard(context, run)

    def CancelAppointment(  # noqa: N802
        self, request: pb.CancelAppointmentRequest, context: grpc.ServicerContext
    ) -> pb.CancelAppointmentResponse:
        def run() -> pb.CancelAppointmentResponse:
            appt = self._store.cancel(request.appointment_id, request.patient_phone)
            log.info("cancelled %s", appt.appointment_id)
            return pb.CancelAppointmentResponse(appointment=_appt_pb(appt))

        return self._guard(context, run)

    def ListAppointments(  # noqa: N802
        self, request: pb.ListAppointmentsRequest, context: grpc.ServicerContext
    ) -> pb.ListAppointmentsResponse:
        def run() -> pb.ListAppointmentsResponse:
            rows = self._store.list_appointments(request.patient_phone, request.include_cancelled)
            return pb.ListAppointmentsResponse(appointments=[_appt_pb(a) for a in rows])

        return self._guard(context, run)


def build_server(store: Store, address: str = "[::]:0") -> tuple[grpc.Server, int]:
    """Create (not start) a server. Port 0 picks a free port; used by the tests."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_BookingServiceServicer_to_server(BookingServicer(store), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    service_name = pb.DESCRIPTOR.services_by_name["BookingService"].full_name
    for name in ("", service_name):
        health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)

    reflection.enable_server_reflection(
        (
            service_name,
            reflection.SERVICE_NAME,
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
        ),
        server,
    )
    port = server.add_insecure_port(address)
    return server, port


def open_store(settings: Settings) -> Store:
    if settings.booking_store == "sqlite":
        return SqliteBookingStore(settings.sqlite_path)
    init_schema_when_ready(settings.database_url)
    return BookingStore(settings.database_url)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    store = open_store(settings)
    server, port = build_server(store, f"[::]:{settings.grpc_port}")
    server.start()
    log.info("booking service listening on %d", port)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    server.stop(grace=5).wait()
    store.close()


if __name__ == "__main__":
    main()
