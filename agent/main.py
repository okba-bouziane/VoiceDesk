"""LiveKit worker: Silero VAD + turn detection + STT -> LangGraph (LLM) -> TTS."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from livekit.agents import AgentServer, AgentSession, JobContext, JobProcess, cli
from livekit.agents.metrics import AgentMetrics
from livekit.plugins import langchain, silero

from agent.conversation import ReceptionAgent
from agent.demo_guard import daily_cap_reached
from agent.metrics import TurnTracker
from agent.providers import build_stt, build_tts, build_turn_handling
from voicedesk.config import Settings, get_settings, load_config
from voicedesk.runtime import build_runtime

log = logging.getLogger("voicedesk.agent")

TRANSCRIPT_TOPIC = "lk.transcription"
METRICS_TOPIC = "turn_metrics"
CALL_LIMIT_NOTICE = "We have reached the time limit of this demo call. Thank you, goodbye."
DAILY_LIMIT_NOTICE = "The daily demo limit has been reached. Please try again tomorrow."
WARNING_SECONDS = 15  # the goodbye is spoken this long before the hard limit


def prewarm(proc: JobProcess) -> None:
    """Load heavy models once per job process, before a room is assigned."""
    from voicedesk.rag import embed_query

    proc.userdata["vad"] = silero.VAD.load()
    embed_query("warm up")


_settings = get_settings()
server = AgentServer(
    setup_fnc=prewarm,
    port=_settings.agent_http_port,
    num_idle_processes=_settings.agent_idle_processes,
    prometheus_port=_settings.agent_metrics_port,
    prometheus_multiproc_dir=_settings.prometheus_multiproc_dir,
)


async def send_to_transcript(ctx: JobContext, text: str) -> None:
    """Show a message in the web client's transcript without speaking it."""
    await ctx.room.local_participant.send_text(text, topic=TRANSCRIPT_TOPIC)


async def refuse_call(ctx: JobContext) -> None:
    await ctx.connect()
    await send_to_transcript(ctx, DAILY_LIMIT_NOTICE)
    await asyncio.sleep(2)  # let the message reach the client before the room closes
    ctx.shutdown(reason="daily cap")


def publish_turn_metrics(ctx: JobContext, session: AgentSession[Any], version: str) -> None:
    """Send each completed turn's latency to the web client over the data channel."""
    tracker = TurnTracker(version)

    @session.on("metrics_collected")
    def _on_metrics(event: Any) -> None:
        metrics: AgentMetrics = event.metrics
        if (turn := tracker.feed(metrics)) is None:
            return
        payload = json.dumps({"prompt_version": version, **turn.as_dict()}).encode()
        asyncio.ensure_future(ctx.room.local_participant.publish_data(payload, topic=METRICS_TOPIC))


def enforce_call_limit(
    ctx: JobContext, session: AgentSession[Any], settings: Settings
) -> asyncio.Task[None]:
    """End long calls so a public demo cannot burn the free provider quotas."""

    async def end_call() -> None:
        await asyncio.sleep(max(settings.max_call_seconds - WARNING_SECONDS, 1))
        if settings.tts_provider == "none":
            await send_to_transcript(ctx, CALL_LIMIT_NOTICE)
        else:
            await session.say(CALL_LIMIT_NOTICE, allow_interruptions=False)
        await asyncio.sleep(1)
        ctx.shutdown(reason="max call duration")

    timer = asyncio.ensure_future(end_call())

    async def cancel_timer() -> None:
        timer.cancel()

    ctx.add_shutdown_callback(cancel_timer)
    return timer


async def greet(
    ctx: JobContext, session: AgentSession[Any], agent: ReceptionAgent, settings: Settings
) -> None:
    if settings.tts_provider == "none":
        # Text-only mode: replies and the greeting go to the transcript as text.
        session.output.set_audio_enabled(False)
        await send_to_transcript(ctx, agent.greeting())
    else:
        await session.say(agent.greeting(), allow_interruptions=True)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    if daily_cap_reached(settings.demo_daily_session_cap, settings.demo_counter_path):
        await refuse_call(ctx)
        return

    runtime = build_runtime(settings)
    ctx.add_shutdown_callback(runtime.close)
    ctx.log_context_fields = {"room": ctx.room.name}

    session: AgentSession[Any] = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=build_stt(settings),
        llm=langchain.LLMAdapter(
            runtime.graph,
            config={"configurable": {"thread_id": ctx.room.name}},
            stream_mode="custom",
        ),
        tts=build_tts(settings),
        turn_handling=build_turn_handling(settings),
    )
    publish_turn_metrics(ctx, session, load_config()["version"])
    if settings.max_call_seconds:
        enforce_call_limit(ctx, session, settings)

    agent = ReceptionAgent()
    await session.start(agent=agent, room=ctx.room)
    await greet(ctx, session, agent, settings)


if __name__ == "__main__":
    cli.run_app(server)
