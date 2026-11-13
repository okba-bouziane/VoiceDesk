"""Latency benchmark: scripted calls, each turn spoken (macOS `say`), transcribed, run through the
conversation graph and spoken back, with the same provider code the agent uses.

Every turn records STT time and time to the first reply token; their sum is the time from the end
of the caller's speech to the first word the TTS can start on (without the endpointing wait).
With --tts every turn is also spoken and timed to the first audio. Groq's free Orpheus TTS allows
100 requests a day and 1,200 tokens a minute, so TTS is benchmarked separately on a paced sample;
a turn whose TTS hit the rate limit is reported as such instead of counted.

    python -m scripts.latency_bench --label baseline --repeats 2
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import platform
import time
from typing import Any

from livekit.agents.utils import http_context

from agent.providers import build_stt, build_tts
from scripts.voice_roundtrip import measure_turn
from voicedesk.config import ROOT, get_settings, load_config
from voicedesk.runtime import build_runtime

OUT_DIR = ROOT / "docs" / "latency"

# Each list is one call; turns follow each other in the same conversation thread.
CALLS: list[list[str]] = [
    ["Hello, what are your opening hours?", "Do you accept the CHIFA card?", "Thank you, goodbye."],
    [
        "I would like to book a dentist appointment on Sunday.",
        "Ten in the morning, please.",
        "My name is Sara Haddad.",
        "My phone number is 0555 123 456.",
        "Yes, that's right.",
    ],
    [
        "How much does a general consultation cost?",
        "Is there parking near the clinic?",
        "Do I need to fast before a blood test?",
    ],
    [
        "Do you have anything free on Monday for general medicine?",
        "Book the first one for me.",
        "Karim Mansouri, 0661 222 333.",
        "Yes, please book it.",
    ],
    [
        "I need to cancel my appointment.",
        "The number is 0555 123 456.",
        "Yes, cancel it.",
    ],
    ["Can I speak to a real person?"],
]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (no interpolation), so every reported value was actually observed."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarise(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
    }


async def bench(label: str, repeats: int, pause: float, speak: bool = False) -> dict[str, Any]:
    settings = get_settings()
    stt, tts = build_stt(settings), build_tts(settings)
    rows: list[dict[str, Any]] = []
    turn = 0
    for rep in range(repeats):
        for call_no, call in enumerate(CALLS):
            runtime = build_runtime(settings)
            thread = f"bench-{label}-{rep}-{call_no}-{time.time_ns()}"
            try:
                for text in call:
                    row = await measure_turn(
                        runtime, stt, tts, turn, text, thread_id=thread, speak=speak
                    )
                    if row["tts_first_audio_s"] is not None:
                        row["speech_to_audio_s"] = round(
                            row["stt_plus_llm_first_token_s"] + row["tts_first_audio_s"], 3
                        )
                    row |= {"call": call_no, "repeat": rep}
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "asked",
                                    "heard",
                                    "stt_s",
                                    "llm_first_token_s",
                                    "tts_first_audio_s",
                                )
                            }
                        ),
                        flush=True,
                    )
                    turn += 1
                    await asyncio.sleep(pause)  # stay under the provider's per-minute token limit
            finally:
                await runtime.close()
    keys = [
        "stt_s",
        "llm_first_token_s",
        "stt_plus_llm_first_token_s",
        "tts_first_audio_s",
        "speech_to_audio_s",
    ]
    return {
        "label": label,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "prompt_version": load_config()["version"],
        "llm_model": settings.llm_model,
        "fast_llm_model": getattr(settings, "fast_llm_model", "") or None,
        "stt_model": settings.stt_model,
        "tts_model": settings.tts_model,
        "machine": platform.platform(),
        "turns": len(rows),
        "summary": {k: summarise(rows, k) for k in keys},
        "rows": rows,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument(
        "--tts", action="store_true", help="also time the first TTS audio of each reply"
    )
    args = ap.parse_args()
    async with http_context.open():
        result = await bench(args.label, args.repeats, args.pause, args.tts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.label}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))
    print("written", out)


if __name__ == "__main__":
    asyncio.run(main())
