from typing import cast

from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> list[Embedding]:
    return cast(
        list[Embedding],
        _get_model()
        .encode(texts, batch_size=EMBEDDING_BATCH_SIZE, normalize_embeddings=True)
        .tolist(),
    )


def embed_query(query: str) -> Embedding:
    return cast(
        Embedding, _get_model().encode([query], normalize_embeddings=True).tolist()[0]
    )
