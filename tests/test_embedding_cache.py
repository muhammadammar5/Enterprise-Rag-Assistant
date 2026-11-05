from langchain_core.embeddings import Embeddings

from app.embeddings import CachedEmbeddings


class CountingEmbeddings(Embeddings):
    """Deterministic fake model that records how many texts it embedded."""

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str):
        return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.calls += 1
        return self._vec(text)


def make(tmp_path, enabled=True):
    base = CountingEmbeddings()
    return base, CachedEmbeddings(base, str(tmp_path / "c.sqlite"), "fake", enabled)


def test_second_call_hits_cache(tmp_path):
    base, cached = make(tmp_path)
    first = cached.embed_documents(["alpha", "beta"])
    second = cached.embed_documents(["alpha", "beta"])
    assert first == second
    assert base.calls == 2          # model ran only once per text
    assert cached.stats["hits"] == 2


def test_duplicates_in_one_batch_embedded_once(tmp_path):
    base, cached = make(tmp_path)
    cached.embed_documents(["same", "same", "same"])
    assert base.calls == 1


def test_cache_persists_across_instances(tmp_path):
    base1, c1 = make(tmp_path)
    c1.embed_documents(["persist me"])
    base2, c2 = make(tmp_path)
    c2.embed_documents(["persist me"])
    assert base2.calls == 0


def test_disabled_cache_always_calls_model(tmp_path):
    base, cached = make(tmp_path, enabled=False)
    cached.embed_documents(["x"])
    cached.embed_documents(["x"])
    assert base.calls == 2


def test_query_and_document_namespaces_are_separate(tmp_path):
    base, cached = make(tmp_path)
    cached.embed_documents(["hello"])
    cached.embed_query("hello")
    assert base.calls == 2
