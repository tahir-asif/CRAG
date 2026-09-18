import chromadb

from app import vector_store
from app.embeddings.embeddings import embed_texts
from app.models import RetrievedChunk


def index_chunks(chunks: list[RetrievedChunk], collection_name: str) -> None:
    collection = vector_store.get_or_create_collection(collection_name)

    # Clear existing data for this repo (fresh re-index everytime)
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [c.content for c in chunks]
    embeddings = embed_texts(texts=texts)

    ids = [f"{c.file_path}:{c.start_line}:{c.end_line}" for c in chunks]
    metadatas: list[chromadb.Metadata] = [
        {
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "chunk_type": c.chunk_type,
            "name": c.name or "",
        }
        for c in chunks
    ]

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
