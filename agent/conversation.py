"""The LiveKit agent. All dialogue logic lives in the LangGraph graph behind LLMAdapter."""

from __future__ import annotations

from livekit.agents import Agent

from voicedesk.config import load_config

GREETING = (
    "Hello, thank you for calling {clinic}. This is a sample demo clinic. "
    "I can book or cancel an appointment, or answer questions about the clinic. How can I help?"
)


class ReceptionAgent(Agent):
    def __init__(self) -> None:
        # The instructions are unused by the graph (its prompts are versioned in
        # config/voicedesk.yaml) but keep the chat context well-formed.
        super().__init__(instructions="Voice receptionist; dialogue is handled by the graph.")

    def greeting(self) -> str:
        return GREETING.format(clinic=load_config()["clinic"]["name"])
