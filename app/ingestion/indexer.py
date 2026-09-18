from app.adapters.embedder import Embedder
from app.adapters.vector_db import VectorStore
from app.models import Chunk


def index_chunks(
    chunks: list[Chunk],
    collection_name: str,
    *,
    store: VectorStore,
    embedder: Embedder,
) -> None:
    """Embed chunks and persist them into the vector store.

    Re-indexing the same collection replaces its contents.
    """
    embeddings = embedder.embed_texts([c.content for c in chunks])
    store.upsert_chunks(collection_name, chunks, embeddings)
