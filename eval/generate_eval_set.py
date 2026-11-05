"""Draft an evaluation set (question + reference answer) from your own PDFs.

    python -m eval.generate_eval_set --n 50

The questions are LLM-generated from randomly sampled passages. REVIEW THE OUTPUT
BY HAND: delete vague or wrong items and fix reference answers. Your accuracy
numbers are only as trustworthy as this file.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from langchain_openai import ChatOpenAI

from app.config import BASE_DIR, get_settings
from app.ingest import load_pdf, split_documents

PROMPT = """You are building a test set for a document Q&A system.
Read the passage and write ONE specific question that can be answered using only
this passage (no vague questions like "what is this about"), plus a short,
factual reference answer.

Return JSON: {{"question": "...", "answer": "..."}}

Passage:
{passage}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(BASE_DIR / "eval" / "eval_questions.json"))
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("Set OPENAI_API_KEY in .env first.")

    pdfs = sorted(Path(settings.data_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {settings.data_dir}")

    pages = [page for pdf in pdfs for page in load_pdf(pdf)]
    chunks = [c for c in split_documents(pages, 1000, 100) if len(c.page_content) > 500]
    random.Random(args.seed).shuffle(chunks)

    llm = ChatOpenAI(
        model=settings.llm_model, temperature=0.2, api_key=settings.openai_api_key
    ).bind(response_format={"type": "json_object"})

    items, seen = [], set()
    for chunk in chunks:
        if len(items) >= args.n:
            break
        try:
            reply = llm.invoke(PROMPT.format(passage=chunk.page_content))
            data = json.loads(reply.content)
            question, answer = data["question"].strip(), data["answer"].strip()
        except Exception:
            continue
        if question in seen:
            continue
        seen.add(question)
        items.append(
            {
                "question": question,
                "reference_answer": answer,
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
            }
        )
        print(f"[{len(items)}/{args.n}] {question}")

    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(items)} questions to {args.out}")
    print("NEXT: open the file and manually review/fix every question and answer.")


if __name__ == "__main__":
    main()
