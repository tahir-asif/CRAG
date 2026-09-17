from rank_bm25 import BM25Okapi

from app.exceptions import RetrievalError
from app.models import ChunkMetadata, RetrievedChunk
from app.vector_store import get_collection


def bm25_search(
    collection_name: str, query: str, top_k: int = 10
) -> list[RetrievedChunk]:
    try:
        collection = get_collection(collection_name)
        all_docs = collection.get(include=["documents", "metadatas"])
    except ValueError as e:
        raise RetrievalError(
            f"Collection '{collection_name}' not found.", status_code=404
        ) from e
    except (RuntimeError, KeyError) as e:
        raise RetrievalError(f"BM25 fetch failed: {e}") from e

    ids = all_docs["ids"]
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    if documents is None or metadatas is None:
        raise RetrievalError("ChromaDB get() returned an unexpected shape.")
    if not ids:
        return []

    try:
        corpus = [_tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
    except (ValueError, TypeError) as e:
        raise RetrievalError(f"BM25 scoring failed: {e}") from e

    ranked = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )[:top_k]

    chunks: list[RetrievedChunk] = []
    for idx in ranked:
        meta = ChunkMetadata.model_validate(metadatas[idx])
        chunks.append(
            RetrievedChunk(
                content=documents[idx],
                file_path=meta.file_path,
                start_line=meta.start_line,
                end_line=meta.end_line,
                chunk_type=meta.chunk_type,
                name=meta.name or None,
                score=float(scores[idx]),
                source="bm25",
            )
        )
    return chunks


def _tokenize(text: str) -> list[str]:
    return text.lower().replace("_", " ").split()
