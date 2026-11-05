"""Benchmark response latency with and without the embedding cache and streaming.

    python -m eval.benchmark_latency --n 20

Reports (milliseconds):
  1. no cache,  blocking   -> total time until the full answer is ready
  2. cache,     blocking   -> same, with the embedding cache enabled and warm
  3. cache,     streaming  -> time to FIRST token (what the user perceives) and total

Be precise when quoting results: the cache mainly removes repeated embedding work
(re-ingestion, repeated questions); streaming mainly cuts time-to-first-token.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from app.config import BASE_DIR, get_settings
from app.rag import RAGPipeline


def summarize(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "mean": round(statistics.mean(values), 1),
        "p50": round(statistics.median(values), 1),
        "p95": round(p95, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(BASE_DIR / "eval" / "eval_questions.json"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default=str(BASE_DIR / "eval" / "latency_results.json"))
    args = parser.parse_args()

    settings = get_settings()
    questions = [
        q["question"]
        for q in json.loads(Path(args.questions).read_text(encoding="utf-8"))
    ][: args.n]

    pipeline = RAGPipeline(settings)
    if pipeline.count() == 0:
        raise SystemExit("Index is empty. Run: python -m app.ingest --dir data/pdfs")

    # Warm-up: load models / open connections so the first measurement isn't skewed.
    pipeline.embeddings.enabled = False
    for q in questions[:2]:
        pipeline.answer(q)

    # 1) No cache, blocking
    pipeline.embeddings.enabled = False
    no_cache = [pipeline.answer(q)["timings"]["total_ms"] for q in questions]

    # 2) Cache, blocking  (first pass fills the query cache, second pass is measured)
    pipeline.embeddings.enabled = True
    for q in questions:
        pipeline.retriever.retrieve(q)
    cached = [pipeline.answer(q)["timings"]["total_ms"] for q in questions]

    # 3) Cache, streaming
    ttft, stream_total = [], []
    for q in questions:
        for event in pipeline.stream(q):
            if event["type"] == "done":
                if event["timings"]["ttft_ms"] is not None:
                    ttft.append(event["timings"]["ttft_ms"])
                stream_total.append(event["timings"]["total_ms"])

    report = {
        "questions": len(questions),
        "no_cache_blocking_total_ms": summarize(no_cache),
        "cache_blocking_total_ms": summarize(cached),
        "cache_streaming_ttft_ms": summarize(ttft),
        "cache_streaming_total_ms": summarize(stream_total),
        "embedding_cache": pipeline.embeddings.stats,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("| Setup | Mean (ms) | p50 (ms) | p95 (ms) |")
    print("|---|---|---|---|")
    for label, key in [
        ("No cache, blocking (total)", "no_cache_blocking_total_ms"),
        ("Cache, blocking (total)", "cache_blocking_total_ms"),
        ("Cache + streaming (time to first token)", "cache_streaming_ttft_ms"),
        ("Cache + streaming (total)", "cache_streaming_total_ms"),
    ]:
        r = report[key]
        print(f"| {label} | {r['mean']} | {r['p50']} | {r['p95']} |")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
