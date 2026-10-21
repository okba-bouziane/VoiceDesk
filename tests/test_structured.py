from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from tests.fakes import ScriptedLLM
from voicedesk.llm import StructuredOutputError, structured_call
from voicedesk.schemas import AgentAction, BookingSlots, Intent, IntentResult, SlotUpdate

MSG = [{"role": "user", "content": "Caller: hello"}]


async def test_valid_json_is_parsed():
    llm = ScriptedLLM()
    llm.raw["route"] = ['{"intent": "clinic_info", "confidence": 0.8}']
    res = await structured_call(llm, "route", IntentResult, MSG)
    assert res.intent is Intent.INFO and len(llm.calls) == 1


async def test_fenced_and_wrapped_json_is_recovered():
    llm = ScriptedLLM()
    llm.raw["route"] = ['Sure!\n```json\n{"intent": "handoff"}\n```']
    assert (await structured_call(llm, "route", IntentResult, MSG)).intent is Intent.HANDOFF


async def test_invalid_output_is_retried_with_the_error_shown():
    llm = ScriptedLLM()
    llm.raw["route"] = ['{"intent": "teleport"}', "not json at all", '{"intent": "confirm"}']
    res = await structured_call(llm, "route", IntentResult, MSG)
    assert res.intent is Intent.CONFIRM and len(llm.calls) == 3
    feedback = llm.calls[1][1][-1]["content"]
    assert "not valid" in feedback and "intent" in feedback


async def test_gives_up_after_the_retry_budget():
    llm = ScriptedLLM()
    llm.raw["route"] = ["nope"] * 5
    with pytest.raises(StructuredOutputError):
        await structured_call(llm, "route", IntentResult, MSG, max_retries=2)
    assert len(llm.calls) == 3


def test_slot_update_validates_and_normalises():
    u = SlotUpdate(
        date="2026-10-25", time="9:30", patient_phone="0555 12-34-56", specialty=" dentistry "
    )
    assert (
        u.time == dt.time(9, 30) and u.patient_phone == "0555123456" and u.specialty == "dentistry"
    )
    with pytest.raises(ValidationError):
        SlotUpdate(patient_phone="12")
    with pytest.raises(ValidationError):
        SlotUpdate(date="next week")
    with pytest.raises(ValidationError):
        SlotUpdate(time="25:99")


def test_booking_slots_merge_keeps_known_values():
    s = BookingSlots(specialty="dentistry").merge(SlotUpdate(date="2026-10-25"))
    s = s.merge(SlotUpdate())
    assert s.specialty == "dentistry" and s.missing() == ["time", "patient_name", "patient_phone"]


def test_agent_action_is_validated_and_markdown_stripped():
    a = AgentAction(kind="answer", speech="**Open** from `9:00`")
    assert a.speech == "Open from 9:00"
    with pytest.raises(ValidationError):
        AgentAction(kind="answer", speech="")
    with pytest.raises(ValidationError):
        AgentAction(kind="fly", speech="hi")  # type: ignore[arg-type]
