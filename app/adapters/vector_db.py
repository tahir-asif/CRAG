import logging
from typing import Protocol, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Embedding, Metadata

from app.config import CHROMA_PATH
from app.models import Chunk, ChunkMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def upsert_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None: ...

    def query_similar(
        self,
        collection: str,
        embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]: ...

    def get_all_chunks(self, collection: str) -> list[Chunk]: ...

    def list_collections(self) -> list[str]: ...

    def delete_collection(self, collection: str) -> None: ...


def build_vector_store() -> VectorStore:
    """Construct the default vector store.

    The only place in the codebase that knows which vector DB is used.
    Swap to Qdrant by changing this function.
    """
    logger.info("Connecting to ChromaDB at %s", CHROMA_PATH)
    return ChromaVectorStore(chromadb.PersistentClient(path=CHROMA_PATH))


class ChromaVectorStore:
    """ChromaDB implementation of VectorStore."""

    def __init__(self, client: ClientAPI):
        self._client = client

    def upsert_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        col = self._client.get_or_create_collection(collection)
        existing = col.get()
        if existing["ids"]:
            col.delete(ids=existing["ids"])

        metadatas: list[Metadata] = [
            ChunkMetadata(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                chunk_type=c.chunk_type,
                name=c.name or "",
            ).model_dump()
            for c in chunks
        ]

        col.add(
            ids=[f"{c.file_path}:{c.start_line}:{c.end_line}" for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=cast(list[Embedding], embeddings),
            metadatas=metadatas,
        )

    def query_similar(
        self,
        collection: str,
        embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        col = self._client.get_collection(collection)
        results = col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        documents = results["documents"]
        metadatas = results["metadatas"]
        distances = results["distances"]

        if documents is None or metadatas is None or distances is None:
            raise RuntimeError("ChromaDB query returned an unexpected shape")

        out: list[RetrievedChunk] = []
        for i in range(len(ids)):
            meta = ChunkMetadata.model_validate(metadatas[0][i])
            out.append(
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
        return out

    def get_all_chunks(self, collection: str) -> list[Chunk]:
        col = self._client.get_collection(collection)
        data = col.get(include=["documents", "metadatas"])

        documents = data["documents"]
        metadatas = data["metadatas"]

        if documents is None or metadatas is None:
            raise RuntimeError("ChromaDB get() returned an unexpected shape")

        out: list[Chunk] = []
        for doc, meta in zip(documents, metadatas):
            parsed = ChunkMetadata.model_validate(meta)
            out.append(
                Chunk(
                    content=doc,
                    file_path=parsed.file_path,
                    start_line=parsed.start_line,
                    end_line=parsed.end_line,
                    chunk_type=parsed.chunk_type,
                    name=parsed.name or None,
                )
            )
        return out

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, collection: str) -> None:
        self._client.delete_collection(collection)
