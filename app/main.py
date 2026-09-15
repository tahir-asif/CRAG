import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.exceptions import LLMError
from app.generation.llm import generate_answer
from app.logging_config import setup_logging
from app.models import (
    Citation,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

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
    logger.info("Ingesting repo: %s", req.repo_url)
    # currently is a "fake" ingest for testing
    repo_name = req.repo_url.rstrip("/").split("/")[-1]
    fake_chunks = [
        RetrievedChunk(
            content=f"# Stub content from {repo_name}\nprint('hello')",
            file_path="stub.py",
            start_line=1,
            end_line=2,
            chunk_type="module",
            name=None,
            score=1.0,
            source="stub",
        ),
        RetrievedChunk(
            content="def authenticate(user, password):\n    return user == 'admin'",
            file_path="auth.py",
            start_line=10,
            end_line=11,
            chunk_type="function",
            name="authenticate",
            score=1.0,
            source="stub",
        ),
    ]
    _STUB_STORE[repo_name] = fake_chunks
    logger.info("Indexed %s: %d chunks", repo_name, len(fake_chunks))
    return IngestResponse(
        repo_name=repo_name,
        files_indexed=1,
        chunks_created=len(fake_chunks),
        status="ok",
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    logger.info("Query: %r (repo=%s)", req.question[:80], req.repo_name)
    if not _STUB_STORE:
        raise HTTPException(400, "No repos ingested. Call /ingest first.")
    repo_name = _resolve_repo(req.repo_name)

    # Chunks
    chunks = _STUB_STORE[repo_name][: req.rerank_top_k]

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
