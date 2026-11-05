"""Measure answer accuracy for a baseline RAG setup vs. each tuning step.

    python -m eval.evaluate                       # run all four configs
    python -m eval.evaluate --configs baseline tuned
    python -m eval.evaluate --limit 10            # quick smoke test

Configs are cumulative, so you can see what each change contributes:
  baseline        large chunks, no overlap, top-3, no reranking, plain prompt
  chunk_tuning    + smaller chunks with overlap
  reranking       + retrieve 10, rerank with a cross-encoder, keep top-4
  tuned           + refined, grounded prompt            <- the shipped default

Accuracy is graded by an LLM judge against your reference answers (temperature 0).
Spot-check some verdicts by hand before quoting the numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from app.config import BASE_DIR, get_settings
from app.embeddings import build_embeddings
from app.ingest import ingest_directory
from app.rag import RAGPipeline

CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": dict(chunk_size=1500, chunk_overlap=0, top_k_retrieve=3, top_k_final=3,
                     use_reranker=False, prompt_version="baseline"),
    "chunk_tuning": dict(chunk_size=800, chunk_overlap=150, top_k_retrieve=3, top_k_final=3,
                         use_reranker=False, prompt_version="baseline"),
    "reranking": dict(chunk_size=800, chunk_overlap=150, top_k_retrieve=10, top_k_final=4,
                      use_reranker=True, prompt_version="baseline"),
    "tuned": dict(chunk_size=800, chunk_overlap=150, top_k_retrieve=10, top_k_final=4,
                  use_reranker=True, prompt_version="refined"),
}

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}
System answer: {candidate}

Does the system answer convey the same key facts as the reference answer, with no
contradictions? Extra detail is fine. A refusal such as "I could not find this" is
INCORRECT. Reply with exactly one word: CORRECT or INCORRECT."""


def judge(llm: ChatOpenAI, question: str, reference: str, candidate: str) -> bool:
    verdict = llm.invoke(
        JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate)
    ).content.strip().upper()
    return verdict.startswith("CORRECT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(BASE_DIR / "eval" / "eval_questions.json"))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(BASE_DIR / "eval" / "results.json"))
    args = parser.parse_args()

    base = get_settings()
    if not base.openai_api_key:
        raise SystemExit("Set OPENAI_API_KEY in .env first.")

    questions: List[Dict[str, str]] = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    print(f"Evaluating {len(questions)} questions across {args.configs}\n")

    embeddings = build_embeddings(base)  # shared model + shared embedding cache
    judge_llm = ChatOpenAI(model=base.llm_model, temperature=0, api_key=base.openai_api_key)

    built_collections = set()
    results: Dict[str, Any] = {}

    for name in args.configs:
        cfg = CONFIGS[name]
        collection = f"eval_c{cfg['chunk_size']}_o{cfg['chunk_overlap']}"
        settings = replace(base, collection=collection, cache_enabled=True, **cfg)

        if collection not in built_collections:  # configs sharing chunking share an index
            print(f"[{name}] building index '{collection}' ...")
            ingest_directory(settings, embeddings, reset=True)
            built_collections.add(collection)

        pipeline = RAGPipeline(settings, embeddings=embeddings)
        correct, rows = 0, []
        for i, item in enumerate(questions, start=1):
            out = pipeline.answer(item["question"])
            ok = judge(judge_llm, item["question"], item["reference_answer"], out["answer"])
            correct += ok
            rows.append({
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "answer": out["answer"],
                "correct": ok,
                "total_ms": out["timings"]["total_ms"],
            })
            print(f"  [{name}] {i}/{len(questions)} {'OK ' if ok else 'MISS'}")

        accuracy = correct / len(questions)
        results[name] = {
            "config": cfg,
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": len(questions),
            "avg_latency_s": round(statistics.mean(r["total_ms"] for r in rows) / 1000, 2),
            "per_question": rows,
        }
        print(f"[{name}] accuracy = {accuracy:.1%} ({correct}/{len(questions)})\n")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("| Config | Accuracy | Correct | Avg latency (s) |")
    print("|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['accuracy']:.1%} | {r['correct']}/{r['total']} | {r['avg_latency_s']} |")
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
