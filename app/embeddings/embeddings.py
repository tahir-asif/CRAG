from typing import cast

from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed_texts(texts: list[str]) -> list[Embedding]:
    """Embed a batch of texts (used at indexing time)."""
    # SentenceTransformers returns ndarray; ChromaDB accepts Sequence[float].
    # The cast bridges their type declarations, which don't quite line up.
    return cast(
        list[Embedding],
        _model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
        ).tolist(),
    )


def embed_query(query: str) -> Embedding:
    """Embed a single query string (used at retrieval time)."""
    return cast(
        Embedding,
        _model.encode([query], normalize_embeddings=True).tolist()[0],
    )


def get_embedding_model() -> SentenceTransformer:
    return _model
