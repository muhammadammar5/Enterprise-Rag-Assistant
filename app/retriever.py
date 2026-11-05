
from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from .config import Settings


class Retriever:
    def __init__(self, vectorstore, settings: Settings) -> None:
        self.vectorstore = vectorstore
        self.settings = settings
        self._reranker = None

    def warmup(self) -> None:
        """Load the reranker up-front so the first query isn't slow."""
        if self.settings.use_reranker:
            self._get_reranker()

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # lazy import

            self._reranker = CrossEncoder(self.settings.reranker_model)
        return self._reranker

    def retrieve(self, query: str) -> List[Document]:
        s = self.settings
        # Pull a wide candidate pool only if we are going to rerank it.
        k = s.top_k_retrieve if s.use_reranker else s.top_k_final
        candidates = self.vectorstore.similarity_search(query, k=k)
        if not candidates:
            return []

        if not s.use_reranker:
            return candidates[: s.top_k_final]

        scores = self._get_reranker().predict(
            [(query, doc.page_content) for doc in candidates]
        )
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        top = []
        for doc, score in ranked[: s.top_k_final]:
            doc.metadata["rerank_score"] = round(float(score), 4)
            top.append(doc)
        return top
