from app.adapters.reranker import Reranker
from app.exceptions import RetrievalError
from app.models import RetrievedChunk


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    *,
    reranker: Reranker,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    try:
        pairs = [(query, c.content) for c in chunks]
        scores = reranker.score(pairs)
    except (RuntimeError, ValueError, TypeError) as e:
        raise RetrievalError(f"Reranker failed: {e}") from e

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    result: list[RetrievedChunk] = []
    for chunk, score in ranked:
        chunk.score = score
        chunk.source = "rerank"
        result.append(chunk)
    return result
