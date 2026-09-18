from app.adapters.embedder import Embedder
from app.adapters.vector_db import VectorStore
from app.exceptions import RetrievalError
from app.models import RetrievedChunk


def vector_search(
    collection_name: str,
    query: str,
    top_k: int,
    *,
    store: VectorStore,
    embedder: Embedder,
) -> list[RetrievedChunk]:
    try:
        return store.query_similar(collection_name, embedder.embed_query(query), top_k)
    except ValueError as e:
        raise RetrievalError(
            f"Collection '{collection_name}' not found.", status_code=404
        ) from e
    except RuntimeError as e:
        raise RetrievalError(f"Vector search failed: {e}") from e
