"""Build STT / TTS / turn-detection components from configuration."""

from __future__ import annotations

from typing import Any

from livekit.agents.voice.turn import TurnHandlingOptions

# Imported at module level so the plugin registers itself and `download-files` fetches its model.
from livekit.plugins.turn_detector.english import EnglishModel

from voicedesk.config import Settings, load_config


def build_stt(s: Settings) -> Any:
    cfg = load_config()["models"]["stt"][s.stt_provider]
    model = s.stt_model or cfg["model"]
    if s.stt_provider == "groq":
        from livekit.plugins import groq

        return groq.STT(model=model, api_key=s.groq_api_key)
    if s.stt_provider == "openai":
        from livekit.plugins import openai

        return openai.STT(model=model, api_key=s.openai_api_key)
    if s.stt_provider == "deepgram":
        from livekit.plugins import deepgram

        return deepgram.STT(model=model, api_key=s.deepgram_api_key)
    raise ValueError(f"unknown STT_PROVIDER {s.stt_provider!r}")


def build_tts(s: Settings) -> Any:
    if s.tts_provider == "none":  # text-only mode: replies are shown as transcript, not spoken
        return None
    cfg = load_config()["models"]["tts"][s.tts_provider]
    model = s.tts_model or cfg["model"]
    voice = s.tts_voice or cfg.get("voice", "")
    if s.tts_provider == "groq":
        from livekit.plugins import groq

        return groq.TTS(model=model, voice=voice, api_key=s.groq_api_key)
    if s.tts_provider == "openai":
        from livekit.plugins import openai

        return openai.TTS(model=model, voice=voice, api_key=s.openai_api_key)
    if s.tts_provider == "cartesia":
        from livekit.plugins import cartesia

        return cartesia.TTS(model=model, api_key=s.cartesia_api_key)
    raise ValueError(f"unknown TTS_PROVIDER {s.tts_provider!r}")


def build_turn_handling(s: Settings) -> TurnHandlingOptions:
    turn = load_config()["turn"]
    handling: Any = {
        "endpointing": {
            "min_delay": turn["endpointing_min_delay"],
            "max_delay": turn["endpointing_max_delay"],
        },
        "interruption": {"enabled": True},
    }
    if s.turn_detection == "model":
        handling["turn_detection"] = EnglishModel()
    else:
        handling["turn_detection"] = "vad"
    return handling
