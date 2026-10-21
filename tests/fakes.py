"""Test doubles. These exist ONLY in tests: the product code never uses them."""

from __future__ import annotations

import datetime as dt
import itertools
import json
from collections.abc import AsyncIterator

from voicedesk.booking_client import AppointmentInfo, SlotInfo, ToolError
from voicedesk.rag import Hit

FAKE_TODAY = dt.date(2026, 10, 21)  # a Wednesday


class ScriptedLLM:
    """Answers per task. `routes` and `extracts` are matched on the last caller line (substring).
    `raw` can force exact raw replies for a task to test invalid-output handling."""

    def __init__(
        self,
        routes: dict[str, str] | None = None,
        extracts: dict[str, dict] | None = None,
        answer: str = "We are open Sunday to Thursday.",
    ) -> None:
        self.routes = routes or {}
        self.extracts = extracts or {}
        self.answer = answer
        self.raw: dict[str, list[str]] = {}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    @staticmethod
    def _last_caller(messages: list[dict[str, str]]) -> str:
        text = messages[-1]["content"]
        lines = [ln for ln in text.splitlines() if ln.startswith("Caller:")]
        return (lines[-1] if lines else text).lower()

    async def complete(self, task: str, messages: list[dict[str, str]], *, json_mode: bool = False):
        self.calls.append((task, messages))
        if self.raw.get(task):
            return self.raw[task].pop(0)
        last = self._last_caller(messages)
        table = self.routes if task == "route" else self.extracts
        for needle, value in table.items():
            if needle.lower() in last:
                if task == "route":
                    return json.dumps({"intent": value, "confidence": 0.95})
                return json.dumps(value)
        return json.dumps({"intent": "other"}) if task == "route" else "{}"

    async def stream(self, task: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append((task, messages))
        for word in self.answer.split(" "):
            yield word + " "


class FakeTools:
    """In-memory booking tools with the same failure modes as the gRPC service."""

    def __init__(self, taken: set[tuple[str, str]] | None = None) -> None:
        self.taken = taken or set()
        self.booked: list[AppointmentInfo] = []
        self.calls: list[str] = []
        self._ids = itertools.count(1)

    async def check_availability(self, specialty: str, date: str, limit: int = 3):
        self.calls.append(f"check_availability:{specialty}:{date}")
        if dt.date.fromisoformat(date).weekday() in (4, 5):
            raise ToolError("INVALID_ARGUMENT", "the clinic is closed on that day")
        times = ["09:00", "09:30", "10:00", "10:30"]
        return [
            SlotInfo("dr-x", "Dr. Sample", specialty, date, t)
            for t in times
            if (date, t) not in self.taken
        ][:limit]

    async def book(self, *, specialty, date, start_time, patient_name, patient_phone, reason):
        self.calls.append(f"book:{specialty}:{date}:{start_time}")
        if (date, start_time) in self.taken:
            raise ToolError("ALREADY_EXISTS", f"{date} {start_time} is no longer available")
        appt = AppointmentInfo(
            f"appt-{next(self._ids)}",
            patient_name,
            "Dr. Sample",
            specialty,
            date,
            start_time,
            "booked",
        )
        self.taken.add((date, start_time))
        self.booked.append(appt)
        return appt

    async def cancel(self, appointment_id: str, patient_phone: str):
        self.calls.append(f"cancel:{appointment_id}")
        for i, a in enumerate(self.booked):
            if a.appointment_id == appointment_id:
                self.booked.pop(i)
                return AppointmentInfo(**{**a.__dict__, "status": "cancelled"})
        raise ToolError("NOT_FOUND", "appointment not found")

    async def list_appointments(self, patient_phone: str):
        self.calls.append("list")
        return list(self.booked)


def fake_search(hits: list[Hit]):
    async def search(question: str) -> list[Hit]:
        return hits

    return search


GOOD_HIT = Hit(
    "opening-hours.md",
    "Opening hours",
    "Open Sunday to Thursday 9:00 to 16:30.",
    score=0.03,
    similarity=0.7,
    fts_rank=1,
)
