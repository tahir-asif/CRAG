import logging

from app.adapters.embedder import Embedder
from app.adapters.reranker import Reranker
from app.adapters.vector_db import VectorStore
from app.exceptions import AmbiguousRepoError, NoReposError, RepoNotFoundError
from app.generation.llm import generate_answer as _run_llm
from app.models import Citation, RetrievedChunk
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import rerank

logger = logging.getLogger(__name__)


def resolve_repo(requested: str | None, *, store: VectorStore) -> str:
    available = store.list_collections()

    if not available:
        raise NoReposError()
    if requested is None:
        if len(available) == 1:
            return available[0]
        raise AmbiguousRepoError(f"Specify repo name. Ingested repos: {available}")

    if requested not in available:
        raise RepoNotFoundError(f"Repo '{requested}' not found. Available: {available}")
    return requested


def retrieve_chunks(
    repo_name: str,
    question: str,
    top_k: int,
    rerank_top_k: int,
    *,
    store: VectorStore,
    embedder: Embedder,
    reranker: Reranker,
) -> list[RetrievedChunk]:
    logger.info(
        "Retrieving %s (top_k=%d, rerank_top_k=%d)",
        repo_name,
        top_k,
        rerank_top_k,
    )
    candidates = hybrid_search(
        repo_name, question, top_k, store=store, embedder=embedder
    )
    return rerank(question, candidates, rerank_top_k, reranker=reranker)


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    api_key: str | None,
) -> str:
    return _run_llm(question, chunks, api_key=api_key)


def generate_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            file_path=c.file_path,
            start_line=c.start_line,
            end_line=c.end_line,
            name=c.name,
        )
        for c in chunks
    ]
