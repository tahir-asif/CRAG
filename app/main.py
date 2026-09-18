import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.exceptions import DomainError
from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo
from app.ingestion.indexer import index_chunks
from app.logging_config import setup_logging
from app.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.utilities.query_util import (
    answer_question,
    generate_citations,
    resolve_repo,
    retrieve_chunks,
)
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


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exception: DomainError):
    if exception.status_code >= 500:
        logger.error("%s: %s", type(exception).__name__, exception)
    else:
        logger.warning("%s: %s", type(exception).__name__, exception)

    return JSONResponse(
        status_code=exception.status_code,
        content={"detail": str(exception)},
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

    repo_path = clone_repo(req.repo_url, req.branch)
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
        files_indexed=files_indexed,
        chunks_created=len(chunks),
        status="ok",
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    logger.info("Query: %r (repo=%s)", req.question[:80], req.repo_name)

    repo_name = resolve_repo(req.repo_name)

    chunks = retrieve_chunks(repo_name, req.question, req.top_k, req.rerank_top_k)
    answer = answer_question(req.question, chunks, req.api_key)
    citations = generate_citations(chunks)

    logger.info("Returned %d citations for %r", len(citations), req.question[:80])
    return QueryResponse(answer=answer, citations=citations, retrieved_chunks=chunks)
