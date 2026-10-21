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

from langchain_core.messages import HumanMessage
from livekit import rtc
from livekit.agents.utils import http_context

from agent.providers import build_stt, build_tts
from voicedesk.config import get_settings
from voicedesk.runtime import build_runtime

DEFAULT = [
    "Hello, what are your opening hours?",
    "Do you accept the CHIFA card?",
    "I would like to book a dentist appointment on Sunday at ten.",
]


async def synth(tts, text: str) -> tuple[rtc.AudioFrame, float]:
    """Returns the audio and the time to first audio byte."""
    start = time.perf_counter()
    first = None
    frames = []
    async for ev in tts.synthesize(text):
        if first is None:
            first = time.perf_counter() - start
        frames.append(ev.frame)
    return rtc.combine_audio_frames(frames), first or 0.0


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


async def main(questions: list[str]) -> None:
    s = get_settings()
    stt, tts = build_stt(s), build_tts(s)
    graph, tools, retriever = build_runtime(s)
    rows = []
    for i, q in enumerate(questions):
        audio = caller_audio(q)
        t0 = time.perf_counter()
        ev = await stt.recognize(audio)
        stt_s = time.perf_counter() - t0
        heard = ev.alternatives[0].text.strip()

        t1 = time.perf_counter()
        first, reply = None, []
        async for chunk in graph.astream(
            {"messages": [HumanMessage(heard, id=f"u{i}")]},
            {"configurable": {"thread_id": "rt"}},
            stream_mode="custom",
        ):
            if first is None:
                first = time.perf_counter() - t1
            reply.append(chunk)
        llm_total = time.perf_counter() - t1
        text = "".join(reply).strip()

        # TTS for the first sentence mimics the streaming pipeline: speech starts after it.
        first_sentence = text.split(". ")[0][:200]
        try:
            _, tts_ttfb = await synth(tts, first_sentence)
        except Exception as err:  # noqa: BLE001
            tts_ttfb = None
            print("TTS unavailable:", str(err)[:120])
        row = {
            "asked": q,
            "heard": heard,
            "reply": text[:160],
            "stt_s": round(stt_s, 3),
            "llm_first_token_s": round(first or 0, 3),
            "llm_total_s": round(llm_total, 3),
            "tts_first_audio_s": None if tts_ttfb is None else round(tts_ttfb, 3),
            "stt_plus_llm_first_token_s": round(stt_s + (first or 0), 3),
        }
        rows.append(row)
        print(json.dumps(row, indent=1))
    pre = [r["stt_plus_llm_first_token_s"] for r in rows]
    print("median STT + LLM first token:", statistics.median(pre), "s over", len(pre), "turns")
    await tools.close()
    retriever.close()


async def run(questions: list[str]) -> None:
    async with http_context.open():
        await main(questions)


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1:] or DEFAULT))
