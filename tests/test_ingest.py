from langchain_core.documents import Document

from app.ingest import chunk_id, split_documents


def make_doc(text, page=1):
    return Document(page_content=text, metadata={"source": "a.pdf", "page": page})


def test_split_respects_chunk_size_and_keeps_metadata():
    text = ("This is a sentence about retrieval. " * 100).strip()
    chunks = split_documents([make_doc(text)], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(c.metadata["source"] == "a.pdf" for c in chunks)


def test_tiny_chunks_are_dropped():
    assert split_documents([make_doc("hi")], 300, 50) == []


def test_chunk_id_is_deterministic_and_content_sensitive():
    a, b = make_doc("same text here " * 5), make_doc("same text here " * 5)
    c = make_doc("different text " * 5)
    assert chunk_id(a) == chunk_id(b)
    assert chunk_id(a) != chunk_id(c)
    assert chunk_id(a) != chunk_id(make_doc("same text here " * 5, page=2))
