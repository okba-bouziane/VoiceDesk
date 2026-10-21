"""Retrieval metrics on the labelled question set (shared by pytest and scripts/eval.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .config import ROOT
from .rag import Mode, Retriever, is_relevant

QUESTIONS = ROOT / "data" / "eval" / "rag_questions.json"


@dataclass
class RagReport:
    mode: str
    k: int
    n: int
    recall_at_k: float
    mrr: float
    abstain_accuracy: float | None = None


def evaluate(retriever: Retriever, mode: Mode, k: int = 3) -> RagReport:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    hits_at_k, rr = 0, 0.0
    for item in data["in_scope"]:
        hits = retriever.search(item["question"], k=k, mode=mode)
        rank = next(
            (
                i
                for i, h in enumerate(hits, 1)
                if h.source == item["source"]
                and item["heading_contains"].lower() in h.heading.lower()
            ),
            None,
        )
        if rank:
            hits_at_k += 1
            rr += 1 / rank
    n = len(data["in_scope"])
    abstain = None
    if mode == "hybrid":
        ok = sum(
            not is_relevant(retriever.search(q, k=k, mode=mode), retriever.min_similarity)
            for q in data["out_of_scope"]
        )
        answered = sum(
            is_relevant(retriever.search(i["question"], k=k, mode=mode), retriever.min_similarity)
            for i in data["in_scope"]
        )
        abstain = (ok + answered) / (len(data["out_of_scope"]) + n)
    return RagReport(mode, k, n, hits_at_k / n, rr / n, abstain)
