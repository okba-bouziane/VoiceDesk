from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AppointmentStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    APPOINTMENT_STATUS_UNSPECIFIED: _ClassVar[AppointmentStatus]
    APPOINTMENT_STATUS_BOOKED: _ClassVar[AppointmentStatus]
    APPOINTMENT_STATUS_CANCELLED: _ClassVar[AppointmentStatus]
APPOINTMENT_STATUS_UNSPECIFIED: AppointmentStatus
APPOINTMENT_STATUS_BOOKED: AppointmentStatus
APPOINTMENT_STATUS_CANCELLED: AppointmentStatus

class Slot(_message.Message):
    __slots__ = ("provider_id", "provider_name", "specialty", "date", "start_time", "duration_minutes")
    PROVIDER_ID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    SPECIALTY_FIELD_NUMBER: _ClassVar[int]
    DATE_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    DURATION_MINUTES_FIELD_NUMBER: _ClassVar[int]
    provider_id: str
    provider_name: str
    specialty: str
    date: str
    start_time: str
    duration_minutes: int
    def __init__(self, provider_id: _Optional[str] = ..., provider_name: _Optional[str] = ..., specialty: _Optional[str] = ..., date: _Optional[str] = ..., start_time: _Optional[str] = ..., duration_minutes: _Optional[int] = ...) -> None: ...

class Appointment(_message.Message):
    __slots__ = ("appointment_id", "patient_name", "patient_phone", "slot", "reason", "status")
    APPOINTMENT_ID_FIELD_NUMBER: _ClassVar[int]
    PATIENT_NAME_FIELD_NUMBER: _ClassVar[int]
    PATIENT_PHONE_FIELD_NUMBER: _ClassVar[int]
    SLOT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    appointment_id: str
    patient_name: str
    patient_phone: str
    slot: Slot
    reason: str
    status: AppointmentStatus
    def __init__(self, appointment_id: _Optional[str] = ..., patient_name: _Optional[str] = ..., patient_phone: _Optional[str] = ..., slot: _Optional[_Union[Slot, _Mapping]] = ..., reason: _Optional[str] = ..., status: _Optional[_Union[AppointmentStatus, str]] = ...) -> None: ...

class CheckAvailabilityRequest(_message.Message):
    __slots__ = ("specialty", "provider_id", "date", "limit")
    SPECIALTY_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_ID_FIELD_NUMBER: _ClassVar[int]
    DATE_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    specialty: str
    provider_id: str
    date: str
    limit: int
    def __init__(self, specialty: _Optional[str] = ..., provider_id: _Optional[str] = ..., date: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class CheckAvailabilityResponse(_message.Message):
    __slots__ = ("slots",)
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    slots: _containers.RepeatedCompositeFieldContainer[Slot]
    def __init__(self, slots: _Optional[_Iterable[_Union[Slot, _Mapping]]] = ...) -> None: ...

class BookAppointmentRequest(_message.Message):
    __slots__ = ("patient_name", "patient_phone", "provider_id", "specialty", "date", "start_time", "reason")
    PATIENT_NAME_FIELD_NUMBER: _ClassVar[int]
    PATIENT_PHONE_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_ID_FIELD_NUMBER: _ClassVar[int]
    SPECIALTY_FIELD_NUMBER: _ClassVar[int]
    DATE_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    patient_name: str
    patient_phone: str
    provider_id: str
    specialty: str
    date: str
    start_time: str
    reason: str
    def __init__(self, patient_name: _Optional[str] = ..., patient_phone: _Optional[str] = ..., provider_id: _Optional[str] = ..., specialty: _Optional[str] = ..., date: _Optional[str] = ..., start_time: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class BookAppointmentResponse(_message.Message):
    __slots__ = ("appointment",)
    APPOINTMENT_FIELD_NUMBER: _ClassVar[int]
    appointment: Appointment
    def __init__(self, appointment: _Optional[_Union[Appointment, _Mapping]] = ...) -> None: ...

class CancelAppointmentRequest(_message.Message):
    __slots__ = ("appointment_id", "patient_phone")
    APPOINTMENT_ID_FIELD_NUMBER: _ClassVar[int]
    PATIENT_PHONE_FIELD_NUMBER: _ClassVar[int]
    appointment_id: str
    patient_phone: str
    def __init__(self, appointment_id: _Optional[str] = ..., patient_phone: _Optional[str] = ...) -> None: ...

class CancelAppointmentResponse(_message.Message):
    __slots__ = ("appointment",)
    APPOINTMENT_FIELD_NUMBER: _ClassVar[int]
    appointment: Appointment
    def __init__(self, appointment: _Optional[_Union[Appointment, _Mapping]] = ...) -> None: ...

class ListAppointmentsRequest(_message.Message):
    __slots__ = ("patient_phone", "include_cancelled")
    PATIENT_PHONE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_CANCELLED_FIELD_NUMBER: _ClassVar[int]
    patient_phone: str
    include_cancelled: bool
    def __init__(self, patient_phone: _Optional[str] = ..., include_cancelled: _Optional[bool] = ...) -> None: ...

class ListAppointmentsResponse(_message.Message):
    __slots__ = ("appointments",)
    APPOINTMENTS_FIELD_NUMBER: _ClassVar[int]
    appointments: _containers.RepeatedCompositeFieldContainer[Appointment]
    def __init__(self, appointments: _Optional[_Iterable[_Union[Appointment, _Mapping]]] = ...) -> None: ...
