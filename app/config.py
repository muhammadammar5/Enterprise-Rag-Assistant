
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Models -----------------------------------------------------------
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Storage ----------------------------------------------------------
    chroma_dir: str = str(BASE_DIR / "chroma_db")
    collection: str = "documents"
    data_dir: str = str(BASE_DIR / "data" / "pdfs")
    cache_path: str = str(BASE_DIR / "cache" / "embeddings.sqlite")
    history_path: str = str(BASE_DIR / "cache" / "history.sqlite")

    # --- Retrieval --------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieve: int = 10   # candidates pulled from ChromaDB
    top_k_final: int = 4       # passages sent to the LLM after reranking
    use_reranker: bool = True
    prompt_version: str = "refined"  # "baseline" or "refined"

    # --- Performance ------------------------------------------------------
    cache_enabled: bool = True


def get_settings() -> Settings:
    """Build Settings from the current environment (re-read on every call)."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        reranker_model=os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "10")),
        top_k_final=int(os.getenv("TOP_K_FINAL", "4")),
        use_reranker=_env_bool("USE_RERANKER", True),
        cache_enabled=_env_bool("CACHE_ENABLED", True),
        prompt_version=os.getenv("PROMPT_VERSION", "refined"),
    )
