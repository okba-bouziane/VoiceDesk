"""Evaluation gate. Offline checks always run; live checks run when an LLM key is configured.

python -m scripts.eval            # exit code 1 when a threshold is missed
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import statistics
import sys
import time
from dataclasses import asdict

import openai
import psycopg

from voicedesk.config import ROOT, get_settings, load_config
from voicedesk.db import init_schema
from voicedesk.evalrag import evaluate
from voicedesk.graph import calendar_text
from voicedesk.llm import StructuredOutputError, build_llm, structured_call
from voicedesk.rag import Retriever, ingest_documents
from voicedesk.schemas import IntentResult, SlotUpdate

OUT = ROOT / "docs" / "eval-results.json"
LIVE_SET = ROOT / "data" / "eval" / "live_set.json"
TODAY = dt.date(2026, 10, 21)


def eval_rag(dsn: str) -> dict:
    init_schema(dsn)
    ingest_documents(dsn)
    retriever = Retriever(dsn)
    try:
        return {m: asdict(evaluate(retriever, m, 3)) for m in ("hybrid", "vector", "fts")} | {
            "hybrid_recall_at_1": evaluate(retriever, "hybrid", 1).recall_at_k
        }
    finally:
        retriever.close()


async def eval_live() -> dict:
    from voicedesk.config import prompt

    llm = build_llm()
    data = json.loads(LIVE_SET.read_text(encoding="utf-8"))
    attempts: list[int] = []
    latencies: list[float] = []

    class Counting:
        """Wraps the client to count how many raw calls each structured call needed."""

        n = 0

        async def complete(self, task, messages, *, json_mode=False):
            Counting.n += 1
            t = time.perf_counter()
            try:
                return await llm.complete(task, messages, json_mode=json_mode)
            finally:
                latencies.append(time.perf_counter() - t)

        def stream(self, task, messages):
            return llm.stream(task, messages)

    client = Counting()
    correct, invalid = 0, 0
    router = prompt("router", flow="none", awaiting=False)
    for item in data["intents"]:
        Counting.n = 0
        try:
            res = await structured_call(
                client,
                "route",
                IntentResult,
                [
                    {"role": "system", "content": router},
                    {"role": "user", "content": f"Caller: {item['text']}"},
                ],
            )
            correct += res.intent.value == item["intent"]
        except StructuredOutputError:
            invalid += 1
        attempts.append(Counting.n)

    fields_total = fields_ok = 0
    system = prompt(
        "extract_booking",
        calendar=calendar_text(TODAY),
        today=TODAY.isoformat(),
        weekday=TODAY.strftime("%A"),
        specialties="general medicine, dentistry, pediatrics, dermatology",
        known="nothing yet",
    )
    for item in data["extracts"]:
        Counting.n = 0
        try:
            update = await structured_call(
                client,
                "extract",
                SlotUpdate,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Caller: {item['text']}"},
                ],
            )
        except StructuredOutputError:
            invalid += 1
            attempts.append(Counting.n)
            fields_total += len(item["expected"])
            continue
        attempts.append(Counting.n)
        got = json.loads(update.model_dump_json())
        for key, want in item["expected"].items():
            fields_total += 1
            fields_ok += str(got.get(key)).lower().removesuffix(":00") == str(want).lower() or (
                str(got.get(key)) == str(want) or str(got.get(key))[:5] == str(want)[:5]
            )
    n_calls = len(attempts)
    return {
        "model": llm.model,
        "intent_accuracy": correct / len(data["intents"]),
        "intent_examples": len(data["intents"]),
        "slot_field_accuracy": fields_ok / fields_total,
        "slot_fields": fields_total,
        "structured_valid_after_retry": 1 - invalid / n_calls,
        "structured_first_try_valid": sum(a == 1 for a in attempts) / n_calls,
        "llm_json_call_latency_s": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3),
            "calls": len(latencies),
        },
    }


def main() -> int:
    settings = get_settings()
    gate = load_config()["eval_gate"]
    failures: list[str] = []
    results: dict = {"prompt_version": load_config()["version"]}

    try:
        results["rag"] = eval_rag(settings.database_url)
    except psycopg.OperationalError as err:
        print(f"Postgres not reachable: {err}")
        return 2
    h = results["rag"]["hybrid"]
    if h["recall_at_k"] < gate["rag_recall_at_3_min"]:
        failures.append(f"RAG recall@3 {h['recall_at_k']:.3f} < {gate['rag_recall_at_3_min']}")
    if h["mrr"] < gate["rag_mrr_min"]:
        failures.append(f"RAG MRR {h['mrr']:.3f} < {gate['rag_mrr_min']}")
    if h["abstain_accuracy"] < gate["rag_abstain_min"]:
        failures.append(f"abstain accuracy {h['abstain_accuracy']:.3f} < {gate['rag_abstain_min']}")

    key = settings.effective_llm_key
    if key and key != "not-needed":
        try:
            results["live"] = asyncio.run(eval_live())
        except openai.RateLimitError as err:
            results["live"] = f"skipped: provider rate limit ({str(err)[:80]})"
        live = results["live"] if isinstance(results["live"], dict) else None
        if live and live["structured_valid_after_retry"] < gate["structured_validity_min"]:
            failures.append(f"structured validity {live['structured_valid_after_retry']:.3f}")
        if live and live["intent_accuracy"] < gate["routing_accuracy_min_live"]:
            failures.append(f"live intent accuracy {live['intent_accuracy']:.3f}")
    else:
        results["live"] = "skipped: no LLM key configured"

    results["passed"] = not failures
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    for f in failures:
        print("FAIL:", f)
    print("EVAL GATE:", "PASSED" if not failures else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
