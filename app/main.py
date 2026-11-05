
from __future__ import annotations

import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .history import HistoryStore
from .ingest import ingest_pdf
from .rag import RAGPipeline

MAX_UPLOAD_MB = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.pipeline = RAGPipeline(settings)  # loads models once at startup
    app.state.history = HistoryStore(settings.history_path)
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Enterprise Document Q&A Assistant",
    description="Retrieval-augmented generation over your PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    timings: Dict[str, Any]


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/health")
def health() -> Dict[str, Any]:
    pipeline: RAGPipeline = app.state.pipeline
    return {
        "status": "ok",
        "indexed_chunks": pipeline.count(),
        "llm_model": pipeline.settings.llm_model,
        "embedding_cache": pipeline.embeddings.stats,
    }


@app.post("/upload")
def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a PDF; it is chunked, embedded and indexed immediately."""
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    settings = app.state.settings
    pipeline: RAGPipeline = app.state.pipeline
    destination = Path(settings.data_dir) / name
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    if destination.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB.")

    try:
        return ingest_pdf(destination, settings, pipeline.embeddings, pipeline.vectorstore)
    except Exception as exc:  # corrupt / encrypted PDFs
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not process PDF: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> Dict[str, Any]:
    """Return the full answer in one response."""
    result = app.state.pipeline.answer(request.question)
    app.state.history.add(
        request.question, result["answer"], result["sources"], result["timings"]["total_ms"]
    )
    return result


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events: sources -> tokens -> done."""

    def event_stream() -> Iterator[str]:
        sources: List[Dict[str, Any]] = []
        for event in app.state.pipeline.stream(request.question):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "done":
                app.state.history.add(
                    request.question,
                    event["answer"],
                    sources,
                    event["timings"]["total_ms"],
                    event["timings"]["ttft_ms"],
                )
            yield _sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)) -> List[Dict[str, Any]]:
    """Most recent questions with answers, sources and latency."""
    return app.state.history.recent(limit)
