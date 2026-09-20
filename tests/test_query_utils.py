import pytest

from app.exceptions import AmbiguousRepoError, NoReposError, RepoNotFoundError
from app.models import Chunk, RetrievedChunk
from app.utilities import query_utils


def _chunk(name: str, content: str | None = None) -> Chunk:
    return Chunk(
        content=content or f"def {name}(): pass",
        file_path=f"{name}.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        name=name,
    )


def test_resolve_repo_returns_only_repo(fake_store):
    fake_store.upsert_chunks("only-repo", [_chunk("a")], [])
    assert query_utils.resolve_repo(None, store=fake_store) == "only-repo"


def test_resolve_repo_raises_when_empty(fake_store):
    with pytest.raises(NoReposError):
        query_utils.resolve_repo(None, store=fake_store)


def test_resolve_repo_raises_for_unknown(fake_store):
    fake_store.upsert_chunks("known", [_chunk("a")], [])
    with pytest.raises(RepoNotFoundError):
        query_utils.resolve_repo("unknown", store=fake_store)


def test_generate_citations_maps_fields():
    chunks = [
        RetrievedChunk(
            content="x",
            file_path="a.py",
            start_line=10,
            end_line=20,
            chunk_type="function",
            name="f",
            score=0.5,
            source="rerank",
        ),
    ]
    citations = query_utils.generate_citations(chunks)

    assert len(citations) == 1
    assert citations[0].file_path == "a.py"
    assert citations[0].start_line == 10
    assert citations[0].end_line == 20
    assert citations[0].name == "f"


def test_retrieve_chunks_pipeline(fake_store, fake_embedder, fake_reranker):
    fake_store.upsert_chunks(
        "repo",
        [
            _chunk("short", content="x"),
            _chunk("long", content="x" * 200),
        ],
        [],
    )

    result = query_utils.retrieve_chunks(
        "repo",
        "query",
        top_k=10,
        rerank_top_k=5,
        store=fake_store,
        embedder=fake_embedder,
        reranker=fake_reranker,
    )

    assert len(result) == 2
    # Reranker scores by length, so "long" must be first
    assert result[0].name == "long"
    assert all(c.source == "rerank" for c in result)


def test_resolve_repo_raises_when_multiple_and_unspecified(fake_store):
    chunk = Chunk(
        content="x",
        file_path="a.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        name="f",
    )
    fake_store.upsert_chunks("repo1", [chunk], [])
    fake_store.upsert_chunks("repo2", [chunk], [])

    with pytest.raises(AmbiguousRepoError):
        query_utils.resolve_repo(None, store=fake_store)
