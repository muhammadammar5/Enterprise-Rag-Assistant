
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_settings

MIN_CHUNK_CHARS = 30
BATCH_SIZE = 128


def load_pdf(path: Path) -> List[Document]:
    """Load one PDF into per-page Documents with clean metadata."""
    from langchain_community.document_loaders import PyPDFLoader  # lazy import

    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        page.metadata = {
            "source": Path(path).name,
            "page": int(page.metadata.get("page", 0)) + 1,  # 1-indexed for humans
        }
    return pages


def split_documents(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]


def chunk_id(chunk: Document) -> str:
    """Deterministic ID so re-ingesting a file updates chunks instead of duplicating."""
    raw = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{chunk.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def get_vectorstore(settings: Settings, embeddings):
    from langchain_chroma import Chroma  # lazy import

    return Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )


def ingest_pdf(
    path: Path, settings: Settings, embeddings, vectorstore=None
) -> Dict[str, object]:
    vectorstore = vectorstore or get_vectorstore(settings, embeddings)
    pages = load_pdf(Path(path))
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)

    unique: Dict[str, Document] = {}
    for chunk in chunks:
        unique[chunk_id(chunk)] = chunk

    ids, docs = list(unique.keys()), list(unique.values())
    for i in range(0, len(docs), BATCH_SIZE):
        vectorstore.add_documents(docs[i : i + BATCH_SIZE], ids=ids[i : i + BATCH_SIZE])

    return {"file": Path(path).name, "pages": len(pages), "chunks": len(docs)}


def ingest_directory(
    settings: Settings,
    embeddings,
    directory: Optional[str] = None,
    reset: bool = False,
) -> List[Dict[str, object]]:
    directory_path = Path(directory or settings.data_dir)
    pdfs = sorted(directory_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {directory_path}")

    vectorstore = get_vectorstore(settings, embeddings)
    if reset:
        try:
            vectorstore.delete_collection()
        except Exception:  # collection may not exist yet
            pass
        vectorstore = get_vectorstore(settings, embeddings)

    results = []
    for pdf in pdfs:
        start = time.perf_counter()
        info = ingest_pdf(pdf, settings, embeddings, vectorstore)
        info["seconds"] = round(time.perf_counter() - start, 2)
        print(f"  {info['file']}: {info['pages']} pages -> {info['chunks']} chunks "
              f"({info['seconds']}s)")
        results.append(info)
    return results


def main() -> None:
    from .embeddings import build_embeddings

    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--dir", default=None, help="Folder containing PDFs")
    parser.add_argument("--reset", action="store_true", help="Wipe the collection first")
    args = parser.parse_args()

    settings = get_settings()
    embeddings = build_embeddings(settings)

    print(f"Ingesting PDFs (chunk_size={settings.chunk_size}, "
          f"overlap={settings.chunk_overlap}) ...")
    start = time.perf_counter()
    results = ingest_directory(settings, embeddings, args.dir, args.reset)
    pages = sum(r["pages"] for r in results)
    chunks = sum(r["chunks"] for r in results)
    print(f"Done: {len(results)} files, {pages} pages, {chunks} chunks "
          f"in {time.perf_counter() - start:.1f}s | cache {embeddings.stats}")


if __name__ == "__main__":
    main()
