from __future__ import annotations

import pytest

from voicedesk.config import load_config
from voicedesk.evalrag import evaluate
from voicedesk.rag import DOCS_DIR, Retriever, chunk_markdown, fuse, ingest_documents


@pytest.fixture(scope="module")
def retriever(dsn):
    n = ingest_documents(dsn)
    assert n >= 25
    r = Retriever(dsn)
    yield r
    r.close()


def test_chunking_splits_on_headings():
    chunks = chunk_markdown(DOCS_DIR / "opening-hours.md")
    headings = [h for _, h, _ in chunks]
    assert "Opening hours and location - Lunch break" in headings
    assert all(src == "opening-hours.md" for src, _, _ in chunks)


def test_rrf_prefers_documents_found_by_both_rankers():
    vec = [(1, "a", "A", "x", 0.9), (2, "b", "B", "y", 0.8), (3, "c", "C", "z", 0.7)]
    fts = [(3, "c", "C", "z", 1.0), (2, "b", "B", "y", 0.5)]
    hits = fuse(vec, fts, k=3)
    assert [h.source for h in hits] == ["c", "b", "a"]
    assert hits[0].similarity == 0.7 and hits[0].fts_rank == 1


def test_hybrid_finds_the_obvious_answer(retriever):
    top = retriever.search("Are you open on Saturdays?", k=1)[0]
    assert top.source == "opening-hours.md"


@pytest.mark.parametrize("mode", ["hybrid", "vector", "fts"])
def test_recall_threshold(retriever, mode):
    gate = load_config()["eval_gate"]
    report = evaluate(retriever, mode, k=3)
    print(report)
    if mode == "hybrid":
        assert report.recall_at_k >= gate["rag_recall_at_3_min"]
        assert report.mrr >= gate["rag_mrr_min"]
        assert report.abstain_accuracy >= gate["rag_abstain_min"]
    else:
        assert report.recall_at_k >= 0.9  # baselines must stay sane too
