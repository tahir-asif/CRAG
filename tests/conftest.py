import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder, get_reranker, get_store
from app.main import app
from tests.fakes import FakeEmbedder, FakeReranker, FakeVectorStore

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


# --- Live test gating -------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="needs --live flag")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def quiet_logs():
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.WARNING)
    yield
    root.setLevel(previous)


@pytest.fixture
def sample_repo_path() -> Path:
    return FIXTURE_REPO


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def client(fake_store, fake_embedder, fake_reranker):
    """FastAPI TestClient with all external dependencies faked.

    Deliberately NOT `with TestClient(app)` — the non-context form skips
    the lifespan, so real MiniLM/cross-encoder models are never loaded.
    """
    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_reranker] = lambda: fake_reranker
    yield TestClient(app)
    app.dependency_overrides.clear()
