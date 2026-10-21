"""Wire the real dependencies (LLM, gRPC booking tools, pgvector retriever) into the graph."""

from __future__ import annotations

import asyncio

from .booking_client import GrpcBookingTools
from .config import Settings, get_settings, load_config
from .graph import Deps, build_graph
from .llm import build_llm
from .rag import Hit, Retriever


def build_runtime(settings: Settings | None = None):  # type: ignore[no-untyped-def]
    settings = settings or get_settings()
    retriever = Retriever(settings.database_url)

    async def search(question: str) -> list[Hit]:
        return await asyncio.to_thread(retriever.search, question)

    tools = GrpcBookingTools(settings.booking_grpc_addr)
    cfg = load_config()
    deps = Deps(
        llm=build_llm(settings),
        tools=tools,
        search=search,
        min_similarity=cfg["rag"]["min_similarity"],
        history=cfg["turn"]["max_history_messages"],
    )
    return build_graph(deps), tools, retriever
