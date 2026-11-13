"""Voice round trip without a microphone: synthetic caller audio (macOS `say`) -> STT ->
LangGraph turn -> TTS, with the same provider code the agent uses. Prints the real latency of
each stage. If the TTS provider refuses (for example Groq Orpheus before its terms are accepted
in the console) the TTS stage is reported as unavailable instead of being invented.

    python -m scripts.voice_roundtrip "What are your opening hours?" ...
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from typing import Any

from langchain_core.messages import HumanMessage
from livekit import rtc
from livekit.agents import stt as lk_stt
from livekit.agents import tts as lk_tts
from livekit.agents.utils import http_context

from agent.providers import build_stt, build_tts
from voicedesk.config import get_settings
from voicedesk.runtime import Runtime, build_runtime

DEFAULT = [
    "Hello, what are your opening hours?",
    "Do you accept the CHIFA card?",
    "I would like to book a dentist appointment on Sunday at ten.",
]


async def synth(tts: lk_tts.TTS, text: str) -> float:
    """Time to the first audio frame of `text`."""
    start = time.perf_counter()
    first: float | None = None
    async for _ in tts.synthesize(text):
        first = first or time.perf_counter() - start
    return first or 0.0


def caller_audio(text: str) -> rtc.AudioFrame:
    """16 kHz mono speech of `text`, produced by the macOS `say` voice."""
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = f"{tmp}/c.aiff", f"{tmp}/c.wav"
        subprocess.run(["say", "-o", aiff, text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav], check=True
        )
        with wave.open(wav, "rb") as w:
            data = w.readframes(w.getnframes())
            return rtc.AudioFrame(data, w.getframerate(), 1, w.getnframes())


async def transcribe(stt: lk_stt.STT, audio: rtc.AudioFrame) -> tuple[str, float]:
    start = time.perf_counter()
    event = await stt.recognize(audio)
    return event.alternatives[0].text.strip(), time.perf_counter() - start


async def run_graph(
    runtime: Runtime, turn: int, text: str, thread_id: str = "rt"
) -> tuple[str, float, float]:
    """Returns the reply, the time to its first token and the time to the whole reply."""
    start = time.perf_counter()
    first: float | None = None
    chunks: list[str] = []
    async for chunk in runtime.graph.astream(
        {"messages": [HumanMessage(text, id=f"u{turn}")]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="custom",
    ):
        first = first or time.perf_counter() - start
        chunks.append(chunk)
    return "".join(chunks).strip(), first or 0.0, time.perf_counter() - start


async def tts_first_audio(tts: lk_tts.TTS, reply: str) -> float | None:
    """TTS of the first sentence mimics the streaming pipeline: speech starts after it."""
    try:
        return await synth(tts, reply.split(". ")[0][:200])
    except Exception as err:  # noqa: BLE001 - any provider failure is reported, not invented
        print("TTS unavailable:", str(err)[:120])
        return None


async def measure_turn(
    runtime: Runtime,
    stt: lk_stt.STT,
    tts: lk_tts.TTS,
    turn: int,
    question: str,
    thread_id: str = "rt",
    speak: bool = True,
) -> dict[str, Any]:
    heard, stt_s = await transcribe(stt, caller_audio(question))
    reply, first_token_s, llm_total_s = await run_graph(runtime, turn, heard, thread_id)
    tts_s = await tts_first_audio(tts, reply) if speak else None
    return {
        "asked": question,
        "heard": heard,
        "reply": reply[:160],
        "stt_s": round(stt_s, 3),
        "llm_first_token_s": round(first_token_s, 3),
        "llm_total_s": round(llm_total_s, 3),
        "tts_first_audio_s": None if tts_s is None else round(tts_s, 3),
        "stt_plus_llm_first_token_s": round(stt_s + first_token_s, 3),
    }


async def main(questions: list[str]) -> None:
    settings = get_settings()
    stt, tts = build_stt(settings), build_tts(settings)
    runtime = build_runtime(settings)
    try:
        rows = []
        for turn, question in enumerate(questions):
            rows.append(await measure_turn(runtime, stt, tts, turn, question))
            print(json.dumps(rows[-1], indent=1))
    finally:
        await runtime.close()
    pre = [row["stt_plus_llm_first_token_s"] for row in rows]
    print("median STT + LLM first token:", statistics.median(pre), "s over", len(pre), "turns")


async def run(questions: list[str]) -> None:
    async with http_context.open():
        await main(questions)


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1:] or DEFAULT))
