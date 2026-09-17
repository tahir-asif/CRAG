import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.exceptions import LLMError, RetrievalError
from app.generation.llm import generate_answer
from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import IngestionError, clone_repo
from app.ingestion.indexer import index_chunks
from app.logging_config import setup_logging
from app.models import (
    Citation,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.vector_store import list_collections

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI & CORS setup
app = FastAPI(title="Codebase RAG API", version="0.1.0")
# Currently in development so CORS allows everything
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def list_repos():
    return {"repos": list_collections()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    logger.info("Ingesting repo: %s (branch=%s)", req.repo_url, req.branch)

    try:
        repo_path = clone_repo(req.repo_url, req.branch)
        chunks = chunk_repo(repo_path, req.file_extensions)
        index_chunks(chunks, repo_path.name)
    except IngestionError as e:
        logger.warning("Ingestion failed for %s: %s", req.repo_url, e)
        raise HTTPException(e.status_code, str(e))

    files_indexed = len({c.file_path for c in chunks})
    logger.info(
        "Indexed %s: %d files, %d chunks",
        repo_path.name,
        files_indexed,
        len(chunks),
    )

    return IngestResponse(
        repo_name=repo_path.name,
        files_indexed=len({c.file_path for c in chunks}),
        chunks_created=len(chunks),
        status="ok",
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    logger.info("Query: %r (repo=%s)", req.question[:80], req.repo_name)

    repo_name = _resolve_repo(req.repo_name)

    chunks = _retrieve_chunks(repo_name, req.question, req.top_k, req.rerank_top_k)
    answer = _generate_answer(req.question, chunks, req.api_key)
    citations = _generate_citations(chunks)

    logger.info("Returned %d citations for %r", len(citations), req.question[:80])
    return QueryResponse(answer=answer, citations=citations, retrieved_chunks=chunks)


def _resolve_repo(requested: str | None) -> str:
    available = list_collections()

    if not available:
        raise HTTPException(400, "No repos ingested. Call /ingest first.")
    if requested is None:
        return available[0]
    if requested not in available:
        raise HTTPException(
            404, f"Repo '{requested}' not found. Available: {available}"
        )

    return requested


def _retrieve_chunks(
    repo_name: str, question: str, top_k: int, rerank_top_k: int
) -> list[RetrievedChunk]:
    try:
        candidates = hybrid_search(repo_name, question, top_k)
        chunks = rerank(question, candidates, rerank_top_k)
    except RetrievalError as e:
        logger.error("Retrieval failed for %s: %s", repo_name, e)
        raise HTTPException(500, "Retrieval pipeline failed.")

    return chunks


def _generate_answer(
    question: str, chunks: list[RetrievedChunk], api_key: str | None
) -> str:
    try:
        return generate_answer(question, chunks, api_key)
    except LLMError as e:
        logger.warning("LLM error: %s", e)
        raise HTTPException(e.status_code, str(e))


def _generate_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            file_path=c.file_path,
            start_line=c.start_line,
            end_line=c.end_line,
            name=c.name,
        )
        for c in chunks
    ]
