from app.models import Chunk, RetrievedChunk


class FakeVectorStore:
    """In-memory VectorStore for tests. No ChromaDB, no disk."""

    def __init__(self):
        self._data: dict[str, list[Chunk]] = {}

    def upsert_chunks(self, collection, chunks, _embeddings):
        self._data[collection] = list(chunks)

    def query_similar(self, collection, _embedding, top_k):
        if collection not in self._data:
            raise ValueError(f"Collection '{collection}' not found")
        return [
            RetrievedChunk(**c.model_dump(), score=1.0, source="vector")
            for c in self._data[collection][:top_k]
        ]

    def get_all_chunks(self, collection):
        if collection not in self._data:
            raise ValueError(f"Collection '{collection}' not found")
        return list(self._data[collection])

    def list_collections(self):
        return list(self._data.keys())

    def delete_collection(self, collection):
        self._data.pop(collection, None)


class FakeEmbedder:
    """Deterministic 384-dim embeddings. Ignores input."""

    def embed_texts(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, _query):
        return [0.1] * 384


class FakeReranker:
    """Deterministic scoring: longer chunks score higher."""

    def score(self, pairs):
        return [float(len(chunk)) for _, chunk in pairs]
