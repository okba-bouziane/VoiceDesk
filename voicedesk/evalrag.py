"""Retrieval metrics on the labelled question set (shared by pytest and scripts/eval.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import ROOT
from .rag import Hit, Mode, SearchIndex, is_relevant

QUESTIONS = ROOT / "data" / "eval" / "rag_questions.json"


@dataclass
class RagReport:
    mode: str
    k: int
    n: int
    recall_at_k: float
    mrr: float
    abstain_accuracy: float | None = None


def _rank_of_expected(hits: list[Hit], item: dict[str, Any]) -> int | None:
    """1-based rank of the labelled chunk among the hits, or None when it is missing."""
    wanted = item["heading_contains"].lower()
    for rank, hit in enumerate(hits, 1):
        if hit.source == item["source"] and wanted in hit.heading.lower():
            return rank
    return None


def evaluate(retriever: SearchIndex, mode: Mode, k: int = 3) -> RagReport:
    """Recall@k and MRR over the in-scope questions. In hybrid mode it also scores the
    answer-or-abstain gate: in-scope questions should pass it, out-of-scope ones should not."""
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    in_scope, out_of_scope = data["in_scope"], data["out_of_scope"]

    found, reciprocal_ranks, answered = 0, 0.0, 0
    for item in in_scope:
        hits = retriever.search(item["question"], k=k, mode=mode)
        if rank := _rank_of_expected(hits, item):
            found += 1
            reciprocal_ranks += 1 / rank
        answered += is_relevant(hits, retriever.min_similarity)

    abstain = None
    if mode == "hybrid":
        abstained = sum(
            not is_relevant(retriever.search(q, k=k, mode=mode), retriever.min_similarity)
            for q in out_of_scope
        )
        abstain = (abstained + answered) / (len(out_of_scope) + len(in_scope))
    n = len(in_scope)
    return RagReport(mode, k, n, found / n, reciprocal_ranks / n, abstain)
