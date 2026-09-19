import logging
from typing import Annotated

from fastapi import Depends, FastAPI

from app.adapters.embedder import Embedder
from app.adapters.reranker import Reranker
from app.adapters.vector_db import VectorStore
from app.dependencies import (
    get_embedder,
    get_reranker,
    get_store,
)
from app.exception_handlers import register_exception_handlers
from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo
from app.ingestion.indexer import index_chunks
from app.lifespan import lifespan
from app.logging_config import setup_logging
from app.middleware import register_middleware
from app.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.utilities.query_utils import (
    answer_question,
    generate_citations,
    resolve_repo,
    retrieve_chunks,
)

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Codebase RAG API", version="0.2.0", lifespan=lifespan)
    register_middleware(app)
    register_exception_handlers(app)
    return app


app = create_app()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def list_repos(store: Annotated[VectorStore, Depends(get_store)]):
    return {"repos": store.list_collections()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    req: IngestRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
):
    logger.info("Ingesting repo: %s (branch=%s)", req.repo_url, req.branch)

    repo_path = clone_repo(req.repo_url, req.branch)
    chunks = chunk_repo(repo_path, req.file_extensions)
    index_chunks(chunks, repo_path.name, store=store, embedder=embedder)

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
def query(
    req: QueryRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
):
    logger.info("Query: %r (repo=%s)", req.question[:80], req.repo_name)

    repo_name = resolve_repo(req.repo_name, store=store)
    chunks = retrieve_chunks(
        repo_name,
        req.question,
        req.top_k,
        req.rerank_top_k,
        store=store,
        embedder=embedder,
        reranker=reranker,
    )
    answer = answer_question(req.question, chunks, req.api_key)
    citations = generate_citations(chunks)

    logger.info("Returned %d citations for %r", len(citations), req.question[:80])
    return QueryResponse(answer=answer, citations=citations, retrieved_chunks=chunks)
