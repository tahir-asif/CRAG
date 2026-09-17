from app.embeddings.embeddings import embed_query
from app.exceptions import RetrievalError
from app.models import ChunkMetadata, RetrievedChunk
from app.vector_store import get_collection


def vector_search(
    collection_name: str, query: str, top_k: int = 10
) -> list[RetrievedChunk]:
    try:
        collection = get_collection(collection_name)
    except ValueError as e:
        raise RetrievalError(
            f"Collection '{collection_name}' not found.", status_code=404
        ) from e

    try:
        results = collection.query(
            query_embeddings=[embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except (ValueError, RuntimeError, KeyError) as e:
        raise RetrievalError(f"Vector search failed: {e}") from e

    ids = results["ids"][0]
    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]

    if documents is None or metadatas is None or distances is None:
        raise RetrievalError("ChromaDB query returned an unexpected shape.")
    if not ids:
        return []

    chunks: list[RetrievedChunk] = []
    for i in range(len(ids)):
        meta = ChunkMetadata.model_validate(metadatas[0][i])
        chunks.append(
            RetrievedChunk(
                content=documents[0][i],
                file_path=meta.file_path,
                start_line=meta.start_line,
                end_line=meta.end_line,
                chunk_type=meta.chunk_type,
                name=meta.name or None,
                score=1.0 - distances[0][i],
                source="vector",
            )
        )
    return chunks
