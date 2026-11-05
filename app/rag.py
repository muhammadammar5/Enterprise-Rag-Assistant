
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings
from .embeddings import build_embeddings
from .ingest import get_vectorstore
from .retriever import Retriever

NOT_FOUND = "I could not find this in the provided documents."


PROMPTS = {
    "baseline": (
        "You are a helpful assistant.",
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:",
    ),
    "refined": (
        "You are an enterprise document assistant. Answer the user's question using "
        "ONLY the numbered context passages provided.\n"
        "Rules:\n"
        f'1. If the answer is not in the context, reply exactly: "{NOT_FOUND}"\n'
        "2. Be concise and specific. Keep exact figures, names and dates from the context.\n"
        "3. Cite the passages you used by number, e.g. [1], [3].\n"
        "4. Never use outside knowledge and never guess.",
        "Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):",
    ),
}


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content blocks
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source')}, p.{d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    seen, sources = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": key[0],
                "page": key[1],
                "snippet": d.page_content[:200].replace("\n", " "),
            }
        )
    return sources


class RAGPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embeddings=None,
        vectorstore=None,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.embeddings = embeddings or build_embeddings(self.settings)
        self.vectorstore = vectorstore or get_vectorstore(self.settings, self.embeddings)
        self.retriever = Retriever(self.vectorstore, self.settings)
        self.retriever.warmup()
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.temperature,
            api_key=self.settings.openai_api_key,
        )

    # ---------------------------------------------------------------- helpers
    def count(self) -> int:
        """Number of chunks currently indexed."""
        return self.vectorstore._collection.count()  # noqa: SLF001

    def _messages(self, question: str, docs: List[Document]):
        system, user = PROMPTS[self.settings.prompt_version]
        return [
            ("system", system),
            ("human", user.format(context=format_context(docs), question=question)),
        ]

    # ------------------------------------------------------------ blocking API
    def answer(self, question: str) -> Dict[str, Any]:
        start = time.perf_counter()
        docs = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - start) * 1000
        if not docs:
            return {
                "answer": "No documents are indexed yet. Upload a PDF first.",
                "sources": [],
                "timings": {"retrieval_ms": round(retrieval_ms, 1), "total_ms": round(retrieval_ms, 1)},
            }

        response = self.llm.invoke(self._messages(question, docs))
        total_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": _to_text(response.content).strip(),
            "sources": format_sources(docs),
            "timings": {
                "retrieval_ms": round(retrieval_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        }

    # ----------------------------------------------------------- streaming API
    def stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """Yield events: sources -> token* -> done (with timings and the full answer)."""
        start = time.perf_counter()
        docs = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - start) * 1000

        yield {"type": "sources", "sources": format_sources(docs)}
        if not docs:
            message = "No documents are indexed yet. Upload a PDF first."
            yield {"type": "token", "text": message}
            yield {"type": "done", "answer": message,
                   "timings": {"retrieval_ms": round(retrieval_ms, 1),
                               "ttft_ms": None, "total_ms": round(retrieval_ms, 1)}}
            return

        parts: List[str] = []
        ttft_ms: Optional[float] = None
        for chunk in self.llm.stream(self._messages(question, docs)):
            text = _to_text(chunk.content)
            if not text:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            parts.append(text)
            yield {"type": "token", "text": text}

        total_ms = (time.perf_counter() - start) * 1000
        yield {
            "type": "done",
            "answer": "".join(parts).strip(),
            "timings": {
                "retrieval_ms": round(retrieval_ms, 1),
                "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                "total_ms": round(total_ms, 1),
            },
        }
