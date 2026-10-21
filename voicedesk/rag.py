"""Clinic knowledge base: chunking, local embeddings (fastembed), pgvector, hybrid search."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastembed import TextEmbedding
from psycopg_pool import ConnectionPool

from .config import ROOT, load_config

DOCS_DIR = ROOT / "data" / "clinic_docs"
Mode = Literal["hybrid", "vector", "fts"]


@dataclass(frozen=True)
class Hit:
    source: str
    heading: str
    content: str
    score: float
    similarity: float | None  # cosine similarity, None when the chunk had no vector hit
    fts_rank: int | None


@lru_cache
def get_embedder() -> TextEmbedding:
    return TextEmbedding(load_config()["models"]["embedding"]["model"])


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.float32) for v in get_embedder().embed(texts)]


def embed_query(text: str) -> np.ndarray:
    prefix = load_config()["models"]["embedding"]["query_prefix"]
    return np.asarray(next(iter(get_embedder().embed([prefix + text]))), dtype=np.float32)


def chunk_markdown(path: Path) -> list[tuple[str, str, str]]:
    """Split one markdown file on level-2 headings: (source, heading, content)."""
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    chunks = []
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        heading, _, body = part.partition("\n")
        body = body.strip()
        if body:
            chunks.append((path.name, f"{title} - {heading.strip()}", body))
    return chunks


def ingest_documents(dsn: str, docs_dir: Path = DOCS_DIR) -> int:
    """Replace the whole index with the current documents. Returns the chunk count."""
    from .db import connect

    chunks = [c for p in sorted(docs_dir.glob("*.md")) for c in chunk_markdown(p)]
    vectors = embed_documents([f"{h}. {b}" for _, h, b in chunks])
    with connect(dsn, vector=True) as conn, conn.transaction():
        conn.execute("TRUNCATE doc_chunks RESTART IDENTITY")
        for (source, heading, body), vec in zip(chunks, vectors, strict=True):
            conn.execute(
                "INSERT INTO doc_chunks (source, heading, content, embedding) "
                "VALUES (%s, %s, %s, %s)",
                (source, heading, body, vec),
            )
    return len(chunks)


_VECTOR_SQL = """
    SELECT id, source, heading, content, 1 - (embedding <=> %(q)s) AS sim
    FROM doc_chunks ORDER BY embedding <=> %(q)s LIMIT %(n)s
"""
# OR semantics: a spoken question rarely contains every keyword of the answer.
_FTS_SQL = """
    WITH query AS (
        SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')::tsquery AS q
    )
    SELECT id, source, heading, content, ts_rank_cd(tsv, q) AS rank
    FROM doc_chunks, query WHERE tsv @@ q ORDER BY rank DESC LIMIT %(n)s
"""


class Retriever:
    def __init__(self, dsn: str) -> None:
        from pgvector.psycopg import register_vector

        self._pool = ConnectionPool(
            dsn, min_size=1, max_size=4, configure=register_vector, open=True
        )
        cfg = load_config()["rag"]
        self.top_k: int = cfg["top_k"]
        self._candidates: int = cfg["candidates"]
        self._rrf_k: int = cfg["rrf_k"]
        self.min_similarity: float = cfg["min_similarity"]

    def close(self) -> None:
        self._pool.close()

    def search(self, question: str, k: int | None = None, mode: Mode = "hybrid") -> list[Hit]:
        k = k or self.top_k
        qvec = embed_query(question) if mode != "fts" else None
        with self._pool.connection() as conn:
            vec_rows = (
                conn.execute(_VECTOR_SQL, {"q": qvec, "n": self._candidates}).fetchall()
                if mode != "fts"
                else []
            )
            fts_rows = (
                conn.execute(_FTS_SQL, {"text": question, "n": self._candidates}).fetchall()
                if mode != "vector"
                else []
            )
        return fuse(vec_rows, fts_rows, k, self._rrf_k)


def fuse(vec_rows: list, fts_rows: list, k: int, rrf_k: int = 60) -> list[Hit]:
    """Reciprocal rank fusion of the vector and full-text rankings."""
    info: dict[int, tuple[str, str, str]] = {}
    sim: dict[int, float] = {}
    fts_rank: dict[int, int] = {}
    score: dict[int, float] = {}
    for rank, (cid, source, heading, content, s) in enumerate(vec_rows, 1):
        info[cid] = (source, heading, content)
        sim[cid] = float(s)
        score[cid] = score.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (cid, source, heading, content, _r) in enumerate(fts_rows, 1):
        info[cid] = (source, heading, content)
        fts_rank[cid] = rank
        score[cid] = score.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(score, key=lambda c: score[c], reverse=True)[:k]
    return [
        Hit(*info[c], score=score[c], similarity=sim.get(c), fts_rank=fts_rank.get(c))
        for c in ordered
    ]


def is_relevant(hits: list[Hit], min_similarity: float) -> bool:
    """Grounding gate: enough vector similarity, or a full-text hit on a strong vector match."""
    if not hits:
        return False
    best = max((h.similarity or 0.0) for h in hits)
    return best >= min_similarity


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "do",
    "you",
    "your",
    "i",
    "my",
    "can",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "for",
    "it",
    "what",
    "how",
    "me",
    "we",
    "be",
    "if",
    "there",
}


def _terms(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class MemoryRetriever:
    """Hosted-demo backend (`RAG_BACKEND=memory`): the same fastembed vectors and the same
    RRF fusion, but held in memory with a small BM25 ranking instead of Postgres + pgvector."""

    def __init__(self, docs_dir: Path = DOCS_DIR) -> None:
        cfg = load_config()["rag"]
        self.top_k: int = cfg["top_k"]
        self._candidates: int = cfg["candidates"]
        self._rrf_k: int = cfg["rrf_k"]
        self.min_similarity: float = cfg["min_similarity"]
        self._chunks = [c for p in sorted(docs_dir.glob("*.md")) for c in chunk_markdown(p)]
        self._vectors = np.stack(embed_documents([f"{h}. {b}" for _, h, b in self._chunks]))
        self._vectors /= np.linalg.norm(self._vectors, axis=1, keepdims=True)
        self._tokens = [_terms(f"{h} {b}") for _, h, b in self._chunks]
        self._avg_len = sum(len(t) for t in self._tokens) / len(self._tokens)
        df: dict[str, int] = {}
        for toks in self._tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        n = len(self._tokens)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def close(self) -> None:
        pass

    def _bm25(self, query: list[str]) -> list[float]:
        k1, b = 1.5, 0.75
        scores = []
        for toks in self._tokens:
            score = 0.0
            for term in set(query):
                tf = toks.count(term)
                if tf:
                    norm = tf + k1 * (1 - b + b * len(toks) / self._avg_len)
                    score += self._idf.get(term, 0.0) * tf * (k1 + 1) / norm
            scores.append(score)
        return scores

    def search(self, question: str, k: int | None = None, mode: Mode = "hybrid") -> list[Hit]:
        k = k or self.top_k
        vec_rows: list = []
        fts_rows: list = []
        if mode != "fts":
            q = embed_query(question)
            sims = self._vectors @ (q / np.linalg.norm(q))
            for i in np.argsort(-sims)[: self._candidates]:
                src, head, body = self._chunks[i]
                vec_rows.append((int(i), src, head, body, float(sims[i])))
        if mode != "vector":
            bm = self._bm25(_terms(question))
            for i in np.argsort(-np.asarray(bm))[: self._candidates]:
                if bm[i] > 0:
                    src, head, body = self._chunks[i]
                    fts_rows.append((int(i), src, head, body, bm[i]))
        return fuse(vec_rows, fts_rows, k, self._rrf_k)


def build_retriever(dsn: str, backend: str = "postgres") -> Retriever | MemoryRetriever:
    return MemoryRetriever() if backend == "memory" else Retriever(dsn)
