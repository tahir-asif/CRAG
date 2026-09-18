from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.adapters.embedder import Embedder, build_embedder
from app.adapters.reranker import Reranker, build_reranker
from app.adapters.vector_db import VectorStore, build_vector_store


@dataclass
class AppDependencies:
    store: VectorStore
    embedder: Embedder
    reranker: Reranker


def build_default_dependencies() -> AppDependencies:
    return AppDependencies(
        store=build_vector_store(),
        embedder=build_embedder(),
        reranker=build_reranker(),
    )


def get_deps(request: Request) -> AppDependencies:
    return request.app.state.deps


Deps = Annotated[AppDependencies, Depends(get_deps)]


def get_store(deps: Deps) -> VectorStore:
    return deps.store


def get_embedder(deps: Deps) -> Embedder:
    return deps.embedder


def get_reranker(deps: Deps) -> Reranker:
    return deps.reranker
