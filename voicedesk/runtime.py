"""Wire the real dependencies (LLM, gRPC booking tools, retriever) into the graph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .booking_client import GrpcBookingTools
from .config import Settings, get_settings, load_config
from .graph import Deps, build_graph
from .llm import build_llm
from .rag import Hit, SearchIndex, build_retriever


@dataclass
class Runtime:
    """A compiled conversation graph plus the resources that must be closed with it."""

    graph: Any  # compiled LangGraph; its generic parameters add no information here
    tools: GrpcBookingTools
    retriever: SearchIndex

    async def close(self) -> None:
        await self.tools.close()
        self.retriever.close()


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or get_settings()
    config = load_config()
    retriever = build_retriever(settings.database_url, settings.rag_backend)

    async def search(question: str) -> list[Hit]:
        # Embedding and the database query are blocking: keep them off the event loop.
        return await asyncio.to_thread(retriever.search, question)

    tools = GrpcBookingTools(settings.booking_grpc_addr)
    deps = Deps(
        llm=build_llm(settings),
        tools=tools,
        search=search,
        min_similarity=config["rag"]["min_similarity"],
        history=config["turn"]["max_history_messages"],
    )
    return Runtime(build_graph(deps), tools, retriever)
