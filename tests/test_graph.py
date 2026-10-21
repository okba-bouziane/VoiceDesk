from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from tests.fakes import FakeTools, ScriptedLLM
from tests.helpers import EXTRACTS, ROUTES, Convo, make_graph
from voicedesk.graph import decide
from voicedesk.schemas import Intent


@pytest.mark.parametrize(
    ("state", "intent", "expected"),
    [
        ({}, Intent.BOOK, "booking"),
        ({}, Intent.CHECK, "booking"),
        ({}, Intent.CANCEL, "manage"),
        ({}, Intent.LIST, "manage"),
        ({}, Intent.INFO, "faq"),
        ({}, Intent.HANDOFF, "handoff"),
        ({}, Intent.SMALLTALK, "smalltalk"),
        ({}, Intent.OTHER, "smalltalk"),
        ({"awaiting": "booking_confirm"}, Intent.CONFIRM, "book"),
        ({"awaiting": "cancel_confirm"}, Intent.CONFIRM, "do_cancel"),
        ({"awaiting": "booking_confirm"}, Intent.DECLINE, "decline_flow"),
        ({"flow": "booking"}, Intent.OTHER, "booking"),  # caller is just giving details
        ({"flow": "cancel"}, Intent.OTHER, "manage"),
        ({"flow": "booking"}, Intent.INFO, "faq"),  # questions are answered mid-flow
        ({"flow": "booking", "awaiting": "booking_confirm"}, Intent.HANDOFF, "handoff"),
        ({}, Intent.CONFIRM, "smalltalk"),  # a stray "yes" with nothing pending
    ],
)
def test_routing_table(state, intent, expected):
    assert decide(state, intent) == expected


async def test_full_booking_with_slot_filling():
    graph, tools, _ = make_graph()
    c = Convo(graph)
    r = await c.say("I want to book a dentist")
    assert "which day" in r.lower()
    r = await c.say("Sunday")
    assert "9:00 AM, 9:30 AM, 10:00 AM" in r  # availability tool offered real slots
    r = await c.say("10:00")
    assert "full name" in r
    r = await c.say("my name is Sample Patient, 0555123456")
    assert "Shall I book it" in r and "Sunday 25 October at 10:00 AM" in r
    assert c.state()["awaiting"] == "booking_confirm"
    r = await c.say("yes")
    assert "is booked" in r
    assert tools.calls[-1] == "book:dentistry:2026-10-25:10:00"
    assert c.state()["flow"] == "none" and c.state()["action"]["kind"] == "booked"


async def test_slot_taken_between_offer_and_booking_asks_for_another_time():
    graph, tools, _ = make_graph(FakeTools(taken={("2026-10-25", "10:00")}))
    c = Convo(graph)
    for line in ["book a dentist", "Sunday", "10:00", "my name is Sample Patient, 0555123456"]:
        await c.say(line)
    r = await c.say("yes")
    assert "no longer available" in r and "another time" in r
    assert c.state()["slots"]["time"] is None and c.state()["slots"]["specialty"] == "dentistry"


async def test_closed_day_is_explained_and_date_is_asked_again():
    graph, _, _ = make_graph()
    c = Convo(graph)
    await c.say("book a dentist")
    r = await c.say("friday")
    assert "closed" in r and "other day" in r


async def test_declining_confirmation_drops_the_flow():
    graph, tools, _ = make_graph()
    c = Convo(graph)
    for line in ["book a dentist", "Sunday", "10:00", "my name is Sample Patient, 0555123456"]:
        await c.say(line)
    r = await c.say("no thanks")
    assert "dropped" in r and not tools.booked
    assert c.state()["slots"] == {}


async def test_cancel_flow_asks_phone_then_confirms_then_cancels():
    tools = FakeTools()
    graph, _, _ = make_graph(tools)
    c = Convo(graph)
    for line in [
        "book a dentist",
        "Sunday",
        "10:00",
        "my name is Sample Patient, 0555123456",
        "yes",
    ]:
        await c.say(line)
    assert len(tools.booked) == 1
    r = await c.say("I want to cancel")  # the scripted extractor knows the phone from this line
    assert "Do you want me to cancel it" in r and c.state()["awaiting"] == "cancel_confirm"
    r = await c.say("yes")
    assert "is cancelled" in r and not tools.booked


async def test_list_appointments():
    tools = FakeTools()
    graph, _, _ = make_graph(tools)
    c = Convo(graph)
    r = await c.say("what are my appointments")
    assert "don't see any" in r
    for line in [
        "book a dentist",
        "Sunday",
        "10:00",
        "my name is Sample Patient, 0555123456",
        "yes",
    ]:
        await c.say(line)
    r = await c.say("what are my appointments")
    assert "You have 1 appointment" in r and "dentistry" in r


async def test_faq_is_grounded_and_streams_tokens():
    graph, _, llm = make_graph()
    c = Convo(graph)
    chunks = []
    async for ch in graph.astream(
        {"messages": [HumanMessage("opening hours?")]},
        c.cfg,
        stream_mode="custom",
    ):
        chunks.append(ch)
    assert len(chunks) > 3  # token by token, not one block
    assert "".join(chunks).strip() == llm.answer
    assert c.state()["action"]["sources"] == ["opening-hours.md"]
    answer_prompt = next(m for t, m in llm.calls if t == "answer")[0]["content"]
    assert "[opening-hours.md]" in answer_prompt  # retrieved context reached the prompt


async def test_faq_without_relevant_context_refuses_instead_of_guessing():
    graph, _, llm = make_graph(hits=[])
    r = await Convo(graph).say("do you do braces?")
    assert "not sure" in r and not any(t == "answer" for t, _ in llm.calls)


async def test_handoff():
    graph, _, _ = make_graph()
    c = Convo(graph)
    r = await c.say("let me talk to a human")
    assert "member of our staff" in r and c.state()["action"]["handoff"] is True


async def test_invalid_router_output_degrades_gracefully():
    llm = ScriptedLLM(ROUTES, EXTRACTS)
    llm.raw["route"] = ["garbage"] * 3
    graph, _, _ = make_graph(llm=llm)
    r = await Convo(graph).say("hello")
    assert r  # falls back to a safe smalltalk reply instead of crashing


class ExplodingLLM(ScriptedLLM):
    async def complete(self, task, messages, *, json_mode=False):
        raise ConnectionError("provider down")


async def test_provider_outage_becomes_an_apology_and_the_session_survives():
    graph, _, _ = make_graph(llm=ExplodingLLM())
    c = Convo(graph)
    r = await c.say("book a dentist")
    assert "technical problem" in r
    assert "technical problem" in await c.say("hello?")  # next turn still works, no crash
