
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import Settings


class CachedEmbeddings(Embeddings):
    def __init__(
        self,
        base: Embeddings,
        db_path: str,
        namespace: str,
        enabled: bool = True,
    ) -> None:
        self.base = base
        self.namespace = namespace
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self._conn.commit()

   
    def _key(self, text: str, kind: str) -> str:
        raw = f"{self.namespace}\x00{kind}\x00{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _lookup(self, keys: List[str]) -> Dict[str, List[float]]:
        found: Dict[str, List[float]] = {}
        with self._lock:
            for i in range(0, len(keys), 500):  # stay under SQLite's variable limit
                batch = keys[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                    batch,
                ).fetchall()
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32).tolist()
        return found

    def _store(self, items: Dict[str, List[float]]) -> None:
        rows = [
            (key, np.asarray(vec, dtype=np.float32).tobytes())
            for key, vec in items.items()
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)", rows
            )
            self._conn.commit()

    def _cached_embed(
        self,
        texts: List[str],
        kind: str,
        compute: Callable[[List[str]], List[List[float]]],
    ) -> List[List[float]]:
        keys = [self._key(t, kind) for t in texts]
        cached = self._lookup(list(set(keys)))

        missing: Dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in cached:
                missing.setdefault(key, text)

        if missing:
            vectors = compute(list(missing.values()))
            fresh = dict(zip(missing.keys(), vectors))
            self._store(fresh)
            cached.update(fresh)

        misses = sum(1 for k in keys if k in missing)
        self.misses += misses
        self.hits += len(keys) - misses
        return [cached[k] for k in keys]

    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.enabled:
            return self.base.embed_documents(texts)
        return self._cached_embed(texts, "doc", self.base.embed_documents)

    def embed_query(self, text: str) -> List[float]:
        if not self.enabled:
            return self.base.embed_query(text)
        compute = lambda ts: [self.base.embed_query(t) for t in ts]  # noqa: E731
        return self._cached_embed([text], "query", compute)[0]

    @property
    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


def build_embeddings(settings: Settings) -> CachedEmbeddings:
    """Load the Hugging Face model and wrap it with the SQLite cache."""
    from langchain_huggingface import HuggingFaceEmbeddings  # lazy: heavy import

    base = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        # Normalised vectors make L2 distance rank identically to cosine similarity.
        encode_kwargs={"normalize_embeddings": True},
    )
    return CachedEmbeddings(
        base,
        db_path=settings.cache_path,
        namespace=f"{settings.embedding_model}|normalized",
        enabled=settings.cache_enabled,
    )
