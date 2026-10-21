"""gRPC server of the booking microservice (health check and reflection included)."""

from __future__ import annotations

import logging
import os
import signal
import time
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from voicedesk.booking.v1 import booking_pb2 as pb
from voicedesk.booking.v1 import booking_pb2_grpc as pb_grpc

from .sqlite_store import SqliteBookingStore
from .store import AppointmentRow, BookingStore, SlotRow
from .validation import ServiceError

log = logging.getLogger("booking_service")

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
    def __init__(self, store: BookingStore | SqliteBookingStore) -> None:
        self._store = store

    def _guard(self, context: grpc.ServicerContext, fn):  # type: ignore[no-untyped-def]
        try:
            return fn()
        except ServiceError as err:
            context.abort(err.code, err.message)
        except Exception:  # noqa: BLE001 - last line of defence, never leak internals
            log.exception("unhandled error")
            context.abort(grpc.StatusCode.INTERNAL, "internal error")

    def CheckAvailability(self, request, context):  # noqa: N802
        def run() -> pb.CheckAvailabilityResponse:
            slots = self._store.check_availability(
                request.specialty, request.provider_id, request.date, request.limit
            )
            return pb.CheckAvailabilityResponse(slots=[_slot_pb(s) for s in slots])

        return self._guard(context, run)

    def BookAppointment(self, request, context):  # noqa: N802
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

    def CancelAppointment(self, request, context):  # noqa: N802
        def run() -> pb.CancelAppointmentResponse:
            appt = self._store.cancel(request.appointment_id, request.patient_phone)
            log.info("cancelled %s", appt.appointment_id)
            return pb.CancelAppointmentResponse(appointment=_appt_pb(appt))

        return self._guard(context, run)

    def ListAppointments(self, request, context):  # noqa: N802
        def run() -> pb.ListAppointmentsResponse:
            rows = self._store.list_appointments(request.patient_phone, request.include_cancelled)
            return pb.ListAppointmentsResponse(appointments=[_appt_pb(a) for a in rows])

        return self._guard(context, run)


def build_server(
    store: BookingStore | SqliteBookingStore, address: str = "[::]:0"
) -> tuple[grpc.Server, int]:
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


def main() -> None:
    from voicedesk.config import get_settings
    from voicedesk.db import init_schema

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    store: BookingStore | SqliteBookingStore
    if os.getenv("BOOKING_STORE", "postgres") == "sqlite":
        store = SqliteBookingStore(os.getenv("SQLITE_PATH", "/tmp/voicedesk.db"))
    else:
        for attempt in range(30):  # wait for Postgres to accept connections
            try:
                init_schema(settings.database_url)
                break
            except Exception as err:  # noqa: BLE001
                log.warning("database not ready (%s), retry %d", err, attempt + 1)
                time.sleep(2)
        else:
            raise SystemExit("database never became ready")
        store = BookingStore(settings.database_url)
    server, port = build_server(store, f"[::]:{os.getenv('GRPC_PORT', '50051')}")
    server.start()
    log.info("booking service listening on %d", port)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        time.sleep(0.5)
    server.stop(grace=5).wait()
    store.close()


if __name__ == "__main__":
    main()
