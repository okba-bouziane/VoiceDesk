"""Text chat with the conversation graph (real LLM, gRPC service and pgvector).

python -m scripts.chat                 # interactive
python -m scripts.chat "hi" "what are your opening hours?"   # scripted turns
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Iterator

from langchain_core.messages import HumanMessage

from voicedesk.runtime import Runtime, build_runtime


def caller_turns(scripted: list[str]) -> Iterator[str]:
    """Scripted turns when given, otherwise lines typed until EOF."""
    if scripted:
        for text in scripted:
            print(f"you> {text}")
            yield text
        return
    while True:
        try:
            yield input("you> ")
        except EOFError:
            return


async def say(runtime: Runtime, text: str) -> None:
    start = time.perf_counter()
    first_token: float | None = None
    print("bot> ", end="", flush=True)
    async for chunk in runtime.graph.astream(
        {"messages": [HumanMessage(text)]},
        {"configurable": {"thread_id": "cli"}},
        stream_mode="custom",
    ):
        first_token = first_token or time.perf_counter() - start
        print(chunk, end="", flush=True)
    total = time.perf_counter() - start
    print(f"\n     [first token {first_token or 0:.2f}s, total {total:.2f}s]")


async def main(scripted: list[str]) -> None:
    runtime = build_runtime()
    try:
        for text in caller_turns(scripted):
            await say(runtime, text)
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
