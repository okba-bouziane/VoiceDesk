from __future__ import annotations

from langchain_core.messages import HumanMessage

from tests.fakes import FAKE_TODAY, GOOD_HIT, FakeTools, ScriptedLLM, fake_search
from voicedesk.graph import Deps, build_graph

ROUTES = {
    "book": "book_appointment",
    "dentist": "book_appointment",
    "name is": "other",
    "yes": "confirm",
    "no thanks": "decline",
    "cancel": "cancel_appointment",
    "my appointments": "list_appointments",
    "opening hours": "clinic_info",
    "braces": "clinic_info",
    "human": "handoff",
    "hello": "smalltalk",
    "friday": "other",
    "sunday": "other",
    "0555": "other",
    "10:00": "other",
    "ten": "other",
}
EXTRACTS = {
    "book a dentist": {"specialty": "dentistry"},
    "sunday": {"date": "2026-10-25"},
    "friday": {"date": "2026-10-23"},
    "10:00": {"time": "10:00"},
    "name is": {"patient_name": "Sample Patient", "patient_phone": "0555123456"},
    "0555": {"patient_phone": "0555123456"},
    "cancel": {"patient_phone": "0555123456"},
    "my appointments": {"patient_phone": "0555123456"},
}


def make_graph(tools=None, llm=None, hits=None):
    tools = tools or FakeTools()
    llm = llm or ScriptedLLM(ROUTES, EXTRACTS)
    deps = Deps(
        llm=llm,
        tools=tools,
        search=fake_search([GOOD_HIT] if hits is None else hits),
        today=lambda: FAKE_TODAY,
    )
    return build_graph(deps), tools, llm


class Convo:
    """Drives the graph turn by turn like the LiveKit adapter does."""

    def __init__(self, graph, thread="t1"):
        self.graph, self.cfg, self.n = graph, {"configurable": {"thread_id": thread}}, 0

    async def say(self, text: str) -> str:
        self.n += 1
        parts = []
        async for chunk in self.graph.astream(
            {"messages": [HumanMessage(text, id=f"u{self.n}")]}, self.cfg, stream_mode="custom"
        ):
            parts.append(chunk)
        return "".join(parts)

    def state(self):
        return self.graph.get_state(self.cfg).values
