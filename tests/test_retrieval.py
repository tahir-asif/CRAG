import pytest

from app.exceptions import RetrievalError
from app.models import Chunk, RetrievedChunk
from app.retrieval.bm25_search import _tokenize, bm25_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import rerank
from app.retrieval.vector_search import vector_search


def _chunk(name: str, content: str) -> Chunk:
    return Chunk(
        content=content,
        file_path=f"{name}.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        name=name,
    )


def test_tokenize_lowercases_and_splits_underscores():
    assert _tokenize("authenticate_User") == ["authenticate", "user"]
    assert _tokenize("parse token") == ["parse", "token"]


def test_vector_search_returns_chunks(fake_store, fake_embedder):
    fake_store.upsert_chunks("repo", [_chunk("f", "def f(): pass")], [])
    result = vector_search("repo", "query", 5, store=fake_store, embedder=fake_embedder)

    assert len(result) == 1
    assert result[0].source == "vector"
    assert result[0].name == "f"


def test_vector_search_raises_for_missing_collection(fake_store, fake_embedder):
    with pytest.raises(RetrievalError) as info:
        vector_search("missing", "query", 5, store=fake_store, embedder=fake_embedder)
    assert info.value.status_code == 404


def test_bm25_ranks_exact_match_higher(fake_store):
    fake_store.upsert_chunks(
        "repo",
        [
            _chunk("authenticate", "def authenticate(): pass"),
            _chunk("unrelated", "x = 1"),
        ],
        [],
    )
    result = bm25_search("repo", "authenticate", 5, store=fake_store)
    assert result
    assert result[0].name == "authenticate"
    assert result[0].source == "bm25"


def test_bm25_empty_collection_returns_empty(fake_store):
    fake_store.upsert_chunks("repo", [], [])
    assert bm25_search("repo", "q", 5, store=fake_store) == []


def test_hybrid_deduplicates_across_retrievers(fake_store, fake_embedder):
    fake_store.upsert_chunks(
        "repo", [_chunk("f", "def f(): pass"), _chunk("g", "def g(): pass")], []
    )
    result = hybrid_search("repo", "def", 10, store=fake_store, embedder=fake_embedder)

    ids = [(c.file_path, c.start_line) for c in result]
    assert len(ids) == len(set(ids)), "duplicates in hybrid output"
    assert all(c.source == "hybrid" for c in result)


def test_rerank_sorts_by_score():
    short = RetrievedChunk(
        content="x",
        file_path="a.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        name="short",
        score=0.0,
        source="hybrid",
    )
    long = RetrievedChunk(
        content="x" * 100,
        file_path="b.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        name="long",
        score=0.0,
        source="hybrid",
    )
    result = rerank("query", [short, long], 5, reranker=_StubReranker())
    assert result[0].name == "long"
    assert result[0].source == "rerank"


def test_rerank_empty_returns_empty(fake_reranker):
    assert rerank("q", [], 5, reranker=fake_reranker) == []


class _StubReranker:
    """Scores by content length — deterministic and inspectable."""

    def score(self, pairs):
        return [float(len(chunk)) for _, chunk in pairs]
