"""Text chat with the conversation graph (real LLM, gRPC service and pgvector).

python -m scripts.chat                 # interactive
python -m scripts.chat "hi" "what are your opening hours?"   # scripted turns
"""

from __future__ import annotations

import asyncio
import sys
import time

from langchain_core.messages import HumanMessage

from voicedesk.runtime import build_runtime


async def main(turns: list[str]) -> None:
    graph, tools, retriever = build_runtime()
    cfg = {"configurable": {"thread_id": "cli"}}
    interactive = not turns
    while True:
        if interactive:
            try:
                text = input("you> ")
            except EOFError:
                break
        elif turns:
            text = turns.pop(0)
            print(f"you> {text}")
        else:
            break
        start = time.perf_counter()
        first = None
        print("bot> ", end="", flush=True)
        async for chunk in graph.astream(
            {"messages": [HumanMessage(text)]}, cfg, stream_mode="custom"
        ):
            if first is None:
                first = time.perf_counter() - start
            print(chunk, end="", flush=True)
        total = time.perf_counter() - start
        print(f"\n     [first token {first or 0:.2f}s, total {total:.2f}s]")
    await tools.close()
    retriever.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
