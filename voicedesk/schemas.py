"""Pydantic models for every structured output the conversation graph relies on."""

from __future__ import annotations

import datetime as dt
import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Intent(StrEnum):
    BOOK = "book_appointment"
    CHECK = "check_availability"
    CANCEL = "cancel_appointment"
    LIST = "list_appointments"
    INFO = "clinic_info"
    CONFIRM = "confirm"
    DECLINE = "decline"
    HANDOFF = "handoff"
    SMALLTALK = "smalltalk"
    OTHER = "other"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


_PHONE_RE = re.compile(r"^\+?\d{8,15}$")


def normalise_phone(value: str) -> str:
    return re.sub(r"[\s\-().]", "", value)


class SlotUpdate(BaseModel):
    """What the caller said in one turn. Every field is optional and validated."""

    specialty: str | None = None
    date: dt.date | None = None
    time: dt.time | None = None
    patient_name: str | None = None
    patient_phone: str | None = None
    reason: str | None = None
    appointment_date: dt.date | None = None

    @field_validator("patient_phone")
    @classmethod
    def _phone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = normalise_phone(v)
        if not _PHONE_RE.match(v):
            raise ValueError("phone must have 8 to 15 digits")
        return v

    @field_validator("specialty", "patient_name", "reason")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("time", mode="before")
    @classmethod
    def _time(cls, v: object) -> object:
        # Accept "9:30" as well as "09:30".
        if isinstance(v, str) and re.fullmatch(r"\d:\d\d", v):
            return "0" + v
        return v


class BookingSlots(BaseModel):
    """Accumulated slots of the booking flow (persisted in the graph state)."""

    specialty: str | None = None
    date: dt.date | None = None
    time: dt.time | None = None
    patient_name: str | None = None
    patient_phone: str | None = None
    reason: str | None = None

    def merge(self, update: SlotUpdate) -> BookingSlots:
        data = self.model_dump()
        for key in data:
            value = getattr(update, key)
            if value is not None:
                data[key] = value
        return BookingSlots(**data)

    def missing(self) -> list[str]:
        order = ["specialty", "date", "time", "patient_name", "patient_phone"]
        return [f for f in order if getattr(self, f) is None]


ActionKind = Literal[
    "answer",
    "ask_slot",
    "offer_slots",
    "confirm_booking",
    "booked",
    "cancel_confirm",
    "cancelled",
    "appointments",
    "handoff",
    "greeting",
    "decline_flow",
    "error",
]


class AgentAction(BaseModel):
    """The validated final action of a turn: what the agent does and says."""

    kind: ActionKind
    speech: str = Field(min_length=1, max_length=600)
    intent: Intent | None = None
    appointment_id: str | None = None
    sources: list[str] = Field(default_factory=list)
    handoff: bool = False

    @field_validator("speech")
    @classmethod
    def _plain_speech(cls, v: str) -> str:
        # Speech goes to a TTS engine: strip markdown that would be read aloud.
        return re.sub(r"[*_`#>]", "", v).strip() or v
