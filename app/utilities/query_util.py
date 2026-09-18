import logging

from app.exceptions import NoReposError, RepoNotFoundError
from app.generation.llm import generate_answer
from app.models import Citation, RetrievedChunk
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.vector_store import list_collections

logger = logging.getLogger(__name__)


def resolve_repo(requested: str | None) -> str:
    """Pick which indexed repo to query.

    Raises:
        NoReposError: no repos have been indexed.
        RepoNotFoundError: a specific repo was requested but not indexed.
    """
    available = list_collections()

    if not available:
        raise NoReposError()
    if requested is None:
        return available[0]
    if requested not in available:
        raise RepoNotFoundError(f"Repo '{requested}' not found. Available: {available}")
    return requested


def retrieve_chunks(
    repo_name: str,
    question: str,
    top_k: int,
    rerank_top_k: int,
) -> list[RetrievedChunk]:
    """Hybrid retrieval followed by cross-encoder reranking.

    Raises RetrievalError (propagated from the retrieval layer).
    """
    logger.info(
        "Retrieving %s (top_k=%d, rerank_top_k=%d)", repo_name, top_k, rerank_top_k
    )
    candidates = hybrid_search(repo_name, question, top_k=top_k)
    return rerank(question, candidates, top_k=rerank_top_k)


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    api_key: str | None,
) -> str:
    """Call the LLM with the retrieved chunks.

    Raises LLMError (propagated from the generation layer).
    """
    return generate_answer(question, chunks, api_key=api_key)


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
