"""Async gRPC client for the booking microservice, wrapped as the graph's tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import grpc

from voicedesk.booking.v1 import booking_pb2 as pb
from voicedesk.booking.v1 import booking_pb2_grpc as pb_grpc

log = logging.getLogger("voicedesk.booking_client")


@dataclass(frozen=True)
class SlotInfo:
    provider_id: str
    provider_name: str
    specialty: str
    date: str
    start_time: str


@dataclass(frozen=True)
class AppointmentInfo:
    appointment_id: str
    patient_name: str
    provider_name: str
    specialty: str
    date: str
    start_time: str
    status: str


class ToolError(Exception):
    """A tool failed in a way the conversation can explain to the caller."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BookingTools(Protocol):
    async def check_availability(
        self, specialty: str, date: str, limit: int = 3
    ) -> list[SlotInfo]: ...

    async def book(
        self,
        *,
        specialty: str,
        date: str,
        start_time: str,
        patient_name: str,
        patient_phone: str,
        reason: str,
    ) -> AppointmentInfo: ...

    async def cancel(self, appointment_id: str, patient_phone: str) -> AppointmentInfo: ...

    async def list_appointments(self, patient_phone: str) -> list[AppointmentInfo]: ...


def _slot(s: pb.Slot) -> SlotInfo:
    return SlotInfo(s.provider_id, s.provider_name, s.specialty, s.date, s.start_time)


def _appt(a: pb.Appointment) -> AppointmentInfo:
    return AppointmentInfo(
        a.appointment_id,
        a.patient_name,
        a.slot.provider_name,
        a.slot.specialty,
        a.slot.date,
        a.slot.start_time,
        pb.AppointmentStatus.Name(a.status).removeprefix("APPOINTMENT_STATUS_").lower(),
    )


class GrpcBookingTools:
    def __init__(self, address: str, timeout: float = 5.0) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = pb_grpc.BookingServiceStub(self._channel)
        self._timeout = timeout

    async def close(self) -> None:
        await self._channel.close()

    async def _call(self, method, request):  # type: ignore[no-untyped-def]
        try:
            return await method(request, timeout=self._timeout)
        except grpc.aio.AioRpcError as err:
            log.info("booking rpc failed: %s %s", err.code().name, err.details())
            raise ToolError(err.code().name, err.details() or "booking service error") from err

    async def check_availability(self, specialty: str, date: str, limit: int = 3) -> list[SlotInfo]:
        resp = await self._call(
            self._stub.CheckAvailability,
            pb.CheckAvailabilityRequest(specialty=specialty, date=date, limit=limit),
        )
        return [_slot(s) for s in resp.slots]

    async def book(
        self,
        *,
        specialty: str,
        date: str,
        start_time: str,
        patient_name: str,
        patient_phone: str,
        reason: str,
    ) -> AppointmentInfo:
        resp = await self._call(
            self._stub.BookAppointment,
            pb.BookAppointmentRequest(
                patient_name=patient_name,
                patient_phone=patient_phone,
                specialty=specialty,
                date=date,
                start_time=start_time,
                reason=reason,
            ),
        )
        return _appt(resp.appointment)

    async def cancel(self, appointment_id: str, patient_phone: str) -> AppointmentInfo:
        resp = await self._call(
            self._stub.CancelAppointment,
            pb.CancelAppointmentRequest(appointment_id=appointment_id, patient_phone=patient_phone),
        )
        return _appt(resp.appointment)

    async def list_appointments(self, patient_phone: str) -> list[AppointmentInfo]:
        resp = await self._call(
            self._stub.ListAppointments, pb.ListAppointmentsRequest(patient_phone=patient_phone)
        )
        return [_appt(a) for a in resp.appointments]
