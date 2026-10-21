"""LiveKit worker: Silero VAD + turn detection + STT -> LangGraph (LLM) -> TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess, cli
from livekit.plugins import langchain, silero

from agent.conversation import ReceptionAgent
from agent.metrics import TurnTracker
from agent.providers import build_stt, build_tts, build_turn_handling
from voicedesk.config import get_settings, load_config
from voicedesk.runtime import build_runtime

log = logging.getLogger("voicedesk.agent")


def prewarm(proc: JobProcess) -> None:
    """Load heavy models once per job process, before a room is assigned."""
    from voicedesk.rag import embed_query

    proc.userdata["vad"] = silero.VAD.load()
    embed_query("warm up")


server = AgentServer(
    setup_fnc=prewarm,
    port=int(os.getenv("AGENT_HTTP_PORT", "8090")),
    num_idle_processes=int(os.getenv("AGENT_IDLE_PROCESSES", "2")),
    prometheus_port=int(os.getenv("AGENT_METRICS_PORT", "9464")),
    prometheus_multiproc_dir=os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/voicedesk-prom"),
)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    graph, tools, retriever = build_runtime(settings)
    version = load_config()["version"]
    ctx.log_context_fields = {"room": ctx.room.name}

    session: AgentSession[Any] = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=build_stt(settings),
        llm=langchain.LLMAdapter(
            graph,
            config={"configurable": {"thread_id": ctx.room.name}},
            stream_mode="custom",
        ),
        tts=build_tts(settings),
        turn_handling=build_turn_handling(settings),
    )

    tracker = TurnTracker(version)

    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:  # type: ignore[no-untyped-def]
        turn = tracker.feed(ev.metrics)
        if turn is not None:
            payload = json.dumps({"prompt_version": version, **turn.as_dict()}).encode()
            asyncio.ensure_future(
                ctx.room.local_participant.publish_data(payload, topic="turn_metrics")
            )

    async def _cleanup() -> None:
        await tools.close()
        retriever.close()

    ctx.add_shutdown_callback(_cleanup)

    max_seconds = int(os.getenv("MAX_CALL_SECONDS", "0"))

    async def _call_timer() -> None:
        """Demo guard: end long calls so a public demo cannot burn the free provider quotas."""
        await asyncio.sleep(max(max_seconds - 15, 1))
        notice = "We have reached the time limit of this demo call. Thank you, goodbye."
        if settings.tts_provider == "none":
            await ctx.room.local_participant.send_text(notice, topic="lk.transcription")
        else:
            await session.say(notice, allow_interruptions=False)
        await asyncio.sleep(1)
        ctx.shutdown(reason="max call duration")

    timer = asyncio.ensure_future(_call_timer()) if max_seconds else None
    if timer:
        ctx.add_shutdown_callback(lambda: timer.cancel())  # type: ignore[arg-type, return-value]

    agent = ReceptionAgent()
    await session.start(agent=agent, room=ctx.room)
    if settings.tts_provider == "none":
        # Text-only mode: no TTS, so replies and the greeting go to the transcript as text.
        session.output.set_audio_enabled(False)
        await ctx.room.local_participant.send_text(agent.greeting(), topic="lk.transcription")
    else:
        await session.say(agent.greeting(), allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
