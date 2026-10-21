"""LiveKit Agents text-mode conversation tests: the real AgentSession and LLMAdapter drive
the LangGraph graph (scripted model underneath, so no network)."""

from __future__ import annotations

import pytest
from livekit.agents import AgentSession
from livekit.plugins import langchain

from agent.conversation import ReceptionAgent
from tests.helpers import make_graph


@pytest.fixture
async def session():
    graph, tools, llm = make_graph()
    adapter = langchain.LLMAdapter(
        graph, config={"configurable": {"thread_id": "room-1"}}, stream_mode="custom"
    )
    async with AgentSession(llm=adapter) as sess:
        await sess.start(ReceptionAgent())
        yield sess, tools


async def test_smalltalk_gets_one_assistant_message(session):
    session, _ = session
    result = await session.run(user_input="hello")
    result.expect.next_event().is_message(role="assistant")
    result.expect.no_more_events()


async def test_booking_over_several_turns(session):
    session, tools = session
    for line in [
        "I want to book a dentist",
        "Sunday",
        "10:00",
        "my name is Sample Patient 0555123456",
    ]:
        result = await session.run(user_input=line)
        result.expect.next_event().is_message(role="assistant")
    result = await session.run(user_input="yes")
    msg = result.expect.next_event().is_message(role="assistant")
    assert "booked" in msg.event().item.text_content
    assert tools.calls[-1].startswith("book:dentistry")


async def test_faq_answer_reaches_the_session(session):
    session, _ = session
    result = await session.run(user_input="what are your opening hours")
    text = result.expect.next_event().is_message(role="assistant").event().item.text_content
    assert "open" in text.lower()
