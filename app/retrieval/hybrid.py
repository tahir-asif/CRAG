from collections import defaultdict

from app.models import RetrievedChunk
from app.retrieval.bm25_search import bm25_search
from app.retrieval.vector_search import vector_search


def hybrid_search(
    collection_name: str, query: str, top_k: int = 10, k: int = 60
) -> list[RetrievedChunk]:
    vec = vector_search(collection_name, query, top_k=top_k * 2)
    bm = bm25_search(collection_name, query, top_k=top_k * 2)

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
    result = []
    for cid, score in ordered:
        c = chunks_by_id[cid]
        c.score = score
        c.source = "hybrid"
        result.append(c)

    return result
