import logging
from typing import Literal

from app.adapters.embedder import Embedder
from app.adapters.reranker import Reranker
from app.adapters.vector_db import VectorStore
from app.models import EvalResult, EvalSummary, RetrievedChunk
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import rerank
from app.retrieval.vector_search import vector_search
from app.utilities.query_utils import get_answer
from scripts.eval.dataset import QAPair
from scripts.eval.metrics import hit_at_k, keyword_coverage, reciprocal_rank

logger = logging.getLogger(__name__)


Variant = Literal["full", "no-rerank", "vector-only"]

FINAL_TOP_K = 10


def _retrieve_for_variant(
    variant: Variant,
    repo_name: str,
    question: str,
    *,
    store: VectorStore,
    embedder: Embedder,
    reranker: Reranker,
) -> list[RetrievedChunk]:
    """Retrieve the top FINAL_TOP_K chunks using the chosen pipeline.

    - vector-only: semantic similarity only
    - no-rerank:   hybrid (vector + BM25 via RRF), no cross-encoder
    - full:        hybrid with FINAL_TOP_K*2 candidates, reranked to FINAL_TOP_K
    """
    if variant == "vector-only":
        return vector_search(
            repo_name,
            question,
            FINAL_TOP_K,
            store=store,
            embedder=embedder,
        )

    candidates = hybrid_search(
        repo_name,
        question,
        FINAL_TOP_K * 2,
        store=store,
        embedder=embedder,
    )

    if variant == "no-rerank":
        return candidates[:FINAL_TOP_K]

    # full: rerank the candidate pool down to the final size
    return rerank(question, candidates, FINAL_TOP_K, reranker=reranker)


def run_eval(
    qa_pairs: list[QAPair],
    repo_name: str,
    *,
    store: VectorStore,
    embedder: Embedder,
    reranker: Reranker,
    generate: bool = False,
    api_key: str | None = None,
    variant: Variant = "full",
) -> EvalSummary:
    results: list[EvalResult] = []

    for i, qa in enumerate(qa_pairs, start=1):
        logger.info(
            "Eval %d/%d [%s]: %s",
            i,
            len(qa_pairs),
            variant,
            qa.question[:70],
        )

        chunks = _retrieve_for_variant(
            variant,
            repo_name,
            qa.question,
            store=store,
            embedder=embedder,
            reranker=reranker,
        )

        cov: float | None = None
        if generate:
            llm_response = get_answer(qa.question, chunks[:5], api_key=api_key)
            cov = keyword_coverage(llm_response.answer, qa.expected_keywords)

        results.append(
            EvalResult(
                question=qa.question,
                hit_at_5=hit_at_k(chunks, qa.expected_files, k=5),
                hit_at_10=hit_at_k(chunks, qa.expected_files, k=10),
                rr=reciprocal_rank(chunks, qa.expected_files),
                keyword_coverage=cov,
            )
        )

    n = len(results)
    covered = [r.keyword_coverage for r in results if r.keyword_coverage is not None]

    return EvalSummary(
        n=n,
        hit_at_5=sum(r.hit_at_5 for r in results) / n,
        hit_at_10=sum(r.hit_at_10 for r in results) / n,
        mrr=sum(r.rr for r in results) / n,
        keyword_coverage=(sum(covered) / len(covered)) if covered else None,
        results=results,
    )
