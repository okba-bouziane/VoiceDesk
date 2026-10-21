"""Conversation tests against the real model. Skipped automatically without an LLM key."""

from __future__ import annotations

import pytest
from livekit.agents import AgentSession
from livekit.plugins import langchain, openai

from agent.conversation import ReceptionAgent
from tests.fakes import FAKE_TODAY, GOOD_HIT, FakeTools, fake_search
from voicedesk.config import get_settings, load_config
from voicedesk.graph import Deps, build_graph
from voicedesk.llm import build_llm

settings = get_settings()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        settings.effective_llm_key in ("", "not-needed"), reason="no LLM key configured"
    ),
]


@pytest.fixture
async def live():
    tools = FakeTools()
    deps = Deps(
        llm=build_llm(), tools=tools, search=fake_search([GOOD_HIT]), today=lambda: FAKE_TODAY
    )
    cfg = load_config()["models"]["llm"][settings.llm_provider]
    judge = openai.LLM(
        # a separate model keeps the judge out of the main model's free-tier token budget
        model="openai/gpt-oss-20b" if settings.llm_provider == "groq" else cfg["model"],
        base_url=settings.llm_base_url or cfg["base_url"],
        api_key=settings.effective_llm_key,
    )
    adapter = langchain.LLMAdapter(
        build_graph(deps), config={"configurable": {"thread_id": "live"}}, stream_mode="custom"
    )
    async with AgentSession(llm=adapter) as sess:
        await sess.start(ReceptionAgent())
        yield sess, tools, judge


async def test_faq_answer_is_grounded_and_judged(live):
    sess, _, judge = live
    result = await sess.run(user_input="What are your opening hours?")
    await (
        result.expect.next_event()
        .is_message(role="assistant")
        .judge(
            judge, intent="States that the clinic is open Sunday to Thursday, about 9:00 to 16:30."
        )
    )


async def test_booking_conversation_reaches_the_booking_tool(live):
    sess, tools, judge = live
    for line in [
        "Hi, I'd like to book a dentist appointment next Sunday at 10 in the morning.",
        "My name is Sample Patient and my number is 0555 12 34 56.",
    ]:
        await sess.run(user_input=line)
    result = await sess.run(user_input="Yes, please book it.")
    await (
        result.expect.next_event()
        .is_message(role="assistant")
        .judge(judge, intent="Confirms that the dentistry appointment has been booked.")
    )
    assert tools.calls[-1] == "book:dentistry:2026-10-25:10:00"


async def test_emergency_is_handed_off(live):
    sess, _, judge = live
    result = await sess.run(user_input="My father has chest pain and cannot breathe!")
    await (
        result.expect.next_event()
        .is_message(role="assistant")
        .judge(
            judge, intent="Passes the caller to staff and mentions calling the emergency number."
        )
    )
