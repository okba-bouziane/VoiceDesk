"""Per-turn latency tracking: aggregate LiveKit metric events into one record per turn,
log it as JSON and export Prometheus histograms (multiprocess-safe)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

from livekit.agents import metrics as lkm
from prometheus_client import Counter, Histogram

log = logging.getLogger("voicedesk.turn")

_BUCKETS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
STAGE_SECONDS = Histogram(
    "voicedesk_turn_stage_seconds",
    "Latency of one stage of a voice turn",
    ["stage", "prompt_version"],
    buckets=_BUCKETS,
)
TURNS = Counter("voicedesk_turns_total", "Completed voice turns", ["prompt_version"])


@dataclass
class TurnLatency:
    speech_id: str
    eou_delay: float = 0.0  # end of speech -> end of turn decided (includes endpointing wait)
    stt_delay: float = 0.0  # end of speech -> final transcript available
    llm_ttft: float = 0.0  # LangGraph + LLM: time to first token
    tts_ttfb: float = 0.0  # time to first audio byte
    _seen: set[str] = field(default_factory=set, repr=False)

    @property
    def end_to_end(self) -> float:
        """Approximate time from the user stopping to the first audio of the answer."""
        return self.eou_delay + self.llm_ttft + self.tts_ttfb

    def as_dict(self) -> dict[str, float | str]:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["end_to_end"] = self.end_to_end
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


class TurnTracker:
    """Feed it LiveKit metric events; it returns a TurnLatency once EOU, LLM and TTS are in."""

    def __init__(self, prompt_version: str) -> None:
        self._turns: dict[str, TurnLatency] = {}
        self._version = prompt_version

    def feed(self, m: object) -> TurnLatency | None:
        speech_id = getattr(m, "speech_id", None)
        if not speech_id:
            return None
        turn = self._turns.setdefault(speech_id, TurnLatency(speech_id))
        if isinstance(m, lkm.EOUMetrics):
            turn.eou_delay = m.end_of_utterance_delay
            turn.stt_delay = m.transcription_delay
            turn._seen.add("eou")
            self._observe("endpointing", turn.eou_delay)
            self._observe("stt", turn.stt_delay)
        elif isinstance(m, lkm.LLMMetrics) and not m.cancelled and m.ttft >= 0:
            turn.llm_ttft = m.ttft
            turn._seen.add("llm")
            self._observe("llm_ttft", turn.llm_ttft)
        elif isinstance(m, lkm.TTSMetrics) and not m.cancelled and "tts" not in turn._seen:
            turn.tts_ttfb = m.ttfb
            turn._seen.add("tts")
            self._observe("tts_ttfb", turn.tts_ttfb)
        else:
            return None
        if {"eou", "llm", "tts"} <= turn._seen:
            del self._turns[speech_id]
            self._record(turn)
            return turn
        return None

    def _observe(self, stage: str, value: float) -> None:
        STAGE_SECONDS.labels(stage, self._version).observe(value)

    def _record(self, t: TurnLatency) -> None:
        self._observe("end_to_end", t.end_to_end)
        TURNS.labels(self._version).inc()
        log.info("turn_latency %s", json.dumps({"prompt_version": self._version, **t.as_dict()}))
