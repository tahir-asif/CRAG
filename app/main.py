import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.exceptions import LLMError
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
from app.retrieval.vector_search import vector_search

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI & CORS setup
app = FastAPI(title="Codebase RAG API", version="0.1.0")
# Currently in development so CORS allows everything
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Global storage
_STUB_STORE: dict[str, list[RetrievedChunk]] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def list_repos():
    return {"repos": list(_STUB_STORE.keys())}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    logger.info("Ingesting repo: %s (branch=%s)", req.repo_url, req.branch)

    try:
        repo_path = clone_repo(req.repo_url, req.branch)
    except IngestionError as e:
        raise HTTPException(e.status_code, str(e))

    chunks = chunk_repo(repo_path, req.file_extensions)
    index_chunks(chunks, repo_path.name)

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
    if not _STUB_STORE:
        raise HTTPException(400, "No repos ingested. Call /ingest first.")
    repo_name = _resolve_repo(req.repo_name)

    # Chunks
    try:
        chunks = vector_search(repo_name, req.question, top_k=req.top_k)
    except ValueError:
        raise HTTPException(404, f"Repo '{repo_name}' not found.")

    # Answer
    try:
        answer = generate_answer(req.question, chunks, api_key=req.api_key)
    except LLMError as e:
        logger.warning("LLM error for query %r: %s", req.question[:80], e)
        raise HTTPException(e.status_code, str(e))

    # Citations
    citations = [
        Citation(
            file_path=c.file_path,
            start_line=c.start_line,
            end_line=c.end_line,
            name=c.name,
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, citations=citations, retrieved_chunks=chunks)


def _resolve_repo(requested: str | None) -> str:
    if requested is None:
        return next(iter(_STUB_STORE))
    if requested not in _STUB_STORE:
        raise HTTPException(
            404,
            f"Repo '{requested}' not found. Available: {list(_STUB_STORE.keys())}",
        )
    return requested
