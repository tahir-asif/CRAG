from rank_bm25 import BM25Okapi

from app.adapters.vector_db import VectorStore
from app.exceptions import RetrievalError
from app.models import RetrievedChunk


def _tokenize(text: str) -> list[str]:
    """Lowercase, treat underscores as spaces, split on whitespace."""
    return text.lower().replace("_", " ").split()


def bm25_search(
    collection_name: str,
    query: str,
    top_k: int,
    *,
    store: VectorStore,
) -> list[RetrievedChunk]:
    try:
        chunks = store.get_all_chunks(collection_name)
    except ValueError as e:
        raise RetrievalError(
            f"Collection '{collection_name}' not found.", status_code=404
        ) from e
    except RuntimeError as e:
        raise RetrievalError(f"BM25 fetch failed: {e}") from e

    if not chunks:
        return []

    try:
        corpus = [_tokenize(c.content) for c in chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
    except (ValueError, TypeError) as e:
        raise RetrievalError(f"BM25 scoring failed: {e}") from e

    ranked = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )[:top_k]

    out: list[RetrievedChunk] = []
    for idx in ranked:
        c = chunks[idx]
        out.append(
            RetrievedChunk(
                content=c.content,
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                chunk_type=c.chunk_type,
                name=c.name,
                score=float(scores[idx]),
                source="bm25",
            )
        )
    return out
