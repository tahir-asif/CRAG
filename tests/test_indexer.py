from app.ingestion.indexer import index_chunks
from app.models import Chunk


def _chunk(name: str) -> Chunk:
    return Chunk(
        content=f"def {name}(): pass",
        file_path=f"{name}.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        name=name,
    )


def test_index_chunks_stores_all(fake_store, fake_embedder):
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    index_chunks(chunks, "repo", store=fake_store, embedder=fake_embedder)

    assert fake_store.list_collections() == ["repo"]
    assert len(fake_store.get_all_chunks("repo")) == 3


def test_reindex_replaces_existing(fake_store, fake_embedder):
    index_chunks(
        [_chunk("a"), _chunk("b")], "repo", store=fake_store, embedder=fake_embedder
    )
    index_chunks([_chunk("a")], "repo", store=fake_store, embedder=fake_embedder)

    assert len(fake_store.get_all_chunks("repo")) == 1


def test_multiple_collections_are_isolated(fake_store, fake_embedder):
    index_chunks([_chunk("a")], "repo1", store=fake_store, embedder=fake_embedder)
    index_chunks([_chunk("b")], "repo2", store=fake_store, embedder=fake_embedder)

    assert sorted(fake_store.list_collections()) == ["repo1", "repo2"]
    assert fake_store.get_all_chunks("repo1")[0].name == "a"
    assert fake_store.get_all_chunks("repo2")[0].name == "b"
