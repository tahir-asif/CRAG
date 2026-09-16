from app.embeddings.embeddings import embed_query
from app.ingestion.indexer import get_collection
from app.models import ChunkMetadata, RetrievedChunk


def vector_search(
    collection_name: str, query: str, top_k: int = 10
) -> list[RetrievedChunk]:
    collection = get_collection(collection_name)
    query_embedding = [embed_query(query)]

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results["distances"]

    if documents is None or metadatas is None or distances is None:
        raise RuntimeError("ChromaDB query returned an unexpected shape")

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
