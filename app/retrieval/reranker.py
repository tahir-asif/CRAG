from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL
from app.exceptions import RetrievalError
from app.models import RetrievedChunk

_reranker = CrossEncoder(RERANKER_MODEL, device="cpu")


def rerank(
    query: str, chunks: list[RetrievedChunk], top_k: int = 5
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    try:
        pairs = [(query, c.content) for c in chunks]
        scores = _reranker.predict(pairs)
    except (RuntimeError, ValueError, TypeError) as e:
        raise RetrievalError(f"Reranker failed: {e}") from e

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]

    result = []
    for chunk, score in ranked:
        chunk.score = float(score)
        chunk.source = "rerank"
        result.append(chunk)

    return result
