from collections import defaultdict

from app.adapters.embedder import Embedder
from app.adapters.vector_db import VectorStore
from app.models import RetrievedChunk
from app.retrieval.bm25_search import bm25_search
from app.retrieval.vector_search import vector_search


def hybrid_search(
    collection_name: str,
    query: str,
    top_k: int,
    *,
    store: VectorStore,
    embedder: Embedder,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse vector and BM25 results with Reciprocal Rank Fusion."""
    vec = vector_search(
        collection_name, query, top_k * 2, store=store, embedder=embedder
    )
    bm = bm25_search(collection_name, query, top_k * 2, store=store)

    rrf_scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(vec):
        cid = f"{chunk.file_path}:{chunk.start_line}"
        rrf_scores[cid] += 1.0 / (k + rank + 1)
        chunks_by_id[cid] = chunk

    for rank, chunk in enumerate(bm):
        cid = f"{chunk.file_path}:{chunk.start_line}"
        rrf_scores[cid] += 1.0 / (k + rank + 1)
        chunks_by_id[cid] = chunk

    ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    result: list[RetrievedChunk] = []
    for cid, score in ordered:
        chunk = chunks_by_id[cid]
        chunk.score = score
        chunk.source = "hybrid"
        result.append(chunk)
    return result
