"""Settings from the environment plus the versioned YAML config."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "voicedesk.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "groq"
    stt_provider: str = "groq"
    tts_provider: str = "groq"
    llm_model: str = ""
    stt_model: str = ""
    tts_model: str = ""
    tts_voice: str = ""
    turn_detection: str = "model"

    groq_api_key: str = ""
    openai_api_key: str = ""
    deepgram_api_key: str = ""
    cartesia_api_key: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    guard_model: str = ""

    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"
    # URL the browser uses; differs from livekit_url inside docker networks.
    livekit_public_url: str = ""

    database_url: str = Field(default="postgresql://voicedesk:voicedesk@localhost:5434/voicedesk")
    booking_grpc_addr: str = "localhost:50051"
    rag_backend: str = "postgres"  # "memory" for the hosted demo (no Postgres)

    @property
    def effective_llm_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        return {"groq": self.groq_api_key, "openai": self.openai_api_key}.get(
            self.llm_provider, "not-needed"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_config(path: str | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def prompt(name: str, **kwargs: Any) -> str:
    cfg = load_config()
    return str(cfg["prompts"][name]).format(clinic=cfg["clinic"]["name"], **kwargs)
