"""Compare models for the JSON steps (intent routing and slot extraction) on the labelled live set.

Runs the same checks as the live part of `scripts/eval.py` once per model and writes
docs/routing-models.json. Needs an LLM key (Groq by default).

    python -m scripts.compare_routing openai/gpt-oss-20b qwen/qwen3.8-27b
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys

from voicedesk.config import ROOT, get_settings, load_config

OUT = ROOT / "docs" / "routing-models.json"


async def run(model: str) -> dict:
    os.environ["LLM_MODEL"] = model
    os.environ["FAST_LLM_MODEL"] = "none"  # every step on `model`
    get_settings.cache_clear()
    from scripts.eval import eval_live

    return await eval_live()


def main(models: list[str]) -> None:
    results = []
    for model in models:
        r = asyncio.run(run(model))
        print(json.dumps(r))
        results.append(r)
    OUT.write_text(
        json.dumps(
            {
                "date": dt.date.today().isoformat(),
                "prompt_version": load_config()["version"],
                "set": "data/eval/live_set.json",
                "results": results,
            },
            indent=2,
        )
    )
    print("written", OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
