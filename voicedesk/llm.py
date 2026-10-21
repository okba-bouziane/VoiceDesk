"""LLM access: a tiny client protocol, an OpenAI-compatible implementation, and
structured-output calls validated with Pydantic and retried on invalid output."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings, load_config

log = logging.getLogger("voicedesk.llm")
Message = dict[str, str]
M = TypeVar("M", bound=BaseModel)


class LLMClient(Protocol):
    """`task` names the graph step (route, extract, answer, ...). Real clients ignore it;
    it exists for logging and so test doubles can answer deterministically per step."""

    async def complete(
        self, task: str, messages: list[Message], *, json_mode: bool = False
    ) -> str: ...

    def stream(self, task: str, messages: list[Message]) -> AsyncIterator[str]: ...


class OpenAICompatClient:
    """Any OpenAI-compatible chat endpoint: Groq, OpenAI, Ollama, vLLM."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "not-needed", timeout=20, max_retries=4
        )
        self.model = model
        self._temperature = temperature
        # gpt-oss models reason before answering; keep it short for voice latency.
        self._extra: dict[str, Any] = (
            {"extra_body": {"reasoning_effort": "low"}} if "gpt-oss" in model else {}
        )

    async def complete(self, task: str, messages: list[Message], *, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            temperature=0.0 if json_mode else self._temperature,  # type: ignore[arg-type]
            max_tokens=600,
            **kwargs,
            **self._extra,
        )
        return resp.choices[0].message.content or ""

    async def stream(self, task: str, messages: list[Message]) -> AsyncIterator[str]:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            temperature=self._temperature,  # type: ignore[arg-type]
            max_tokens=500,
            stream=True,
            **self._extra,
        )
        async for chunk in cast(Any, resp):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def build_llm(settings: Settings | None = None) -> OpenAICompatClient:
    settings = settings or get_settings()
    cfg = load_config()["models"]["llm"][settings.llm_provider]
    return OpenAICompatClient(
        base_url=settings.llm_base_url or cfg["base_url"],
        api_key=settings.effective_llm_key,
        model=settings.llm_model or cfg["model"],
    )


class StructuredOutputError(RuntimeError):
    """The model never produced a valid object within the retry budget."""


def _extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Models sometimes wrap the object in prose: take the outermost braces.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


async def structured_call(
    llm: LLMClient,
    task: str,
    schema: type[M],
    messages: list[Message],
    *,
    max_retries: int = 2,
) -> M:
    """Ask for JSON, validate it against `schema`; on failure show the model its error and retry."""
    convo = list(messages)
    last_error = ""
    for attempt in range(max_retries + 1):
        raw = await llm.complete(task, convo, json_mode=True)
        try:
            return schema.model_validate(_extract_json(raw))
        except (json.JSONDecodeError, ValidationError) as err:
            last_error = str(err)[:400]
            log.warning("invalid %s output (attempt %d): %s", task, attempt + 1, last_error)
            convo = [
                *messages,
                {"role": "assistant", "content": raw[:600]},
                {
                    "role": "user",
                    "content": "That reply was not valid. Fix it and answer with the JSON "
                    f"object only. Error: {last_error}",
                },
            ]
    raise StructuredOutputError(f"{task}: no valid output after {max_retries + 1} attempts")
