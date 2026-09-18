from typing import Protocol

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


def build_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode([query], normalize_embeddings=True).tolist()[0]
