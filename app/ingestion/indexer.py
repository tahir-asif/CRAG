import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PATH, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from app.models import RetrievedChunk

_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
_client = chromadb.PersistentClient(path=CHROMA_PATH)


def index_chunks(chunks: list[RetrievedChunk], collection_name: str) -> None:
    collection = _client.get_or_create_collection(collection_name)

    # Clear existing data for this repo (fresh re-index everytime)
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [c.content for c in chunks]
    embeddings = _model.encode(
        texts, batch_size=EMBEDDING_BATCH_SIZE, normalize_embeddings=True
    ).tolist()

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


def get_collection(name: str):
    return _client.get_collection(name)


def list_collections() -> list[str]:
    return [c.name for c in _client.list_collections()]
